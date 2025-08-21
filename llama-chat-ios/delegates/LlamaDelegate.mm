//
//  LlamaDelegate.mm
//  llama-chat-objc
//
//  Created by George MacKay-Shore on 19/08/2025.
//

#import <Foundation/Foundation.h>
#include <vector>
#include "LlamaDelegate.hh"

@implementation LlamaDelegate {
    // We don't use @property for these std::vector variables as in-place
    // mutations didn't work correctly.
    std::vector<llama_chat_message> messages;
    std::vector<llama_token> promptTokens;
    
    // The current number of tokens being processed
    int32_t numCurrentTokens;
    
    // The built-in raw chat template
    const char *tmpl;
    
    // The next token to be processed
    llama_token nextTokenId;
}

// Properties for the model, context, vocabulary, sampler, and batch
@synthesize model;
@synthesize context;
@synthesize vocab;
@synthesize sampler;
@synthesize batch;

/// Creates a `LlamaDelegate` instance using the supplied model name.
///
/// The model can be any `gguf` model stored within the bundle. This
/// method also initialises the `llama.cpp` framework, and sets up the model
/// context. If we're running on the simulator, then we force CPU usage for
/// the app.
///
/// @param modelName The name of the model to load (without file extension).
/// @return A new `LlamaDelegate` instance, or `nil` if the model could not be loaded.
///
+ (instancetype) newDelegateWithModelName:(NSString *)modelName {
    auto pathName = [NSBundle.mainBundle pathForResource:modelName ofType:@"gguf"];
    if (!pathName) {
        NSLog(@"Model file not found.");
        return nil;
    }
    
    ggml_backend_load_all();
    auto modelParams = llama_model_default_params();
#if TARGET_OS_SIMULATOR
    modelParams.n_gpu_layers = 0;
    NSLog(@"forcing n_gpu_layers to 0 on simulator");
#endif
    
    auto model = llama_model_load_from_file(pathName.UTF8String, modelParams);
    if (!model) {
        NSLog(@"failed to load model from file: %@", pathName);
        return nil;
    }
    
    auto numThreads = MAX(1, MIN(8, (uint)[NSProcessInfo processInfo].processorCount - 2));
    NSLog(@"using %d threads for inference", numThreads);
    
    auto ctxParams = llama_context_default_params();
    ctxParams.n_threads = numThreads;
    ctxParams.n_threads_batch = numThreads;
    ctxParams.n_ctx = 2048;
    ctxParams.n_batch = 512;
    
    auto context = llama_init_from_model(model, ctxParams);
    if (!context) {
        NSLog(@"failed to create context from model");
        llama_model_free(model);
        return nil;
    }
    
    return [[LlamaDelegate alloc] initWithModel: model andContext:context];
}

/// Initialises a `LlamaDelegate` instance with the specified `llama_model`
/// and `llama_context`.
///
/// This method shouldn't be called directly – but who's gonna stop you?
///
/// We set up the vocabulary and sampler chain, and initialise the list of chat
/// messages (i.e. `{role, content}` pairs). We also initialise the list of
/// prompt tokens, grab a reference to the Jinja chat template for the model,
/// and initialise the token counts.
///
/// @param model The `llama_model` to use for inference.
/// @param context The `llama_context` to use for inference.
/// @return A new `LlamaDelegate` instance, or `nil` if something goes wrong
/// (errors will be logged)
- (instancetype) initWithModel:(llama_model *)model andContext:(llama_context *)context {
    if (!model || !context) {
        NSLog(@"model or context is nil, cannot initialise LlamaDelegate.");
        if (context) {
            llama_free(context);
        }
        if (model) {
            llama_model_free(model);
        }
        return nil;
    }
    
    self = [super init];
    if (!self) {
        llama_free(context);
        llama_model_free(model);
        return nil;
    }
    self.model = model;
    self.context = context;

    self.vocab = llama_model_get_vocab(model);

    auto sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.3f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    self.sampler = sampler;
    
    self->messages = std::vector<llama_chat_message>();
    self->promptTokens = std::vector<llama_token>();
    self->tmpl = llama_model_chat_template(self.model, nullptr);
    self->numCurrentTokens = 0;
    self->nextTokenId = 0;
    
    return self;
}

/// Initialises the completion process for a given prompt.
///
/// Here we tokenise the initial prompt, checking that we have enough context
/// space in which to decode the tokens for sampling. We reference the tokens
/// in a new `llama_batch` and decode that batch ready for sampling, and
/// keep track of the number of tokens we're about to process.
///
/// @param prompt The prompt to infer against
/// @return a `BOOL` indicating success.
- (BOOL) initialiseCompletionForPrompt:(NSString *)prompt {
    if (prompt.length == 0) {
        NSLog(@"prompt is empty, cannot initialise completion.");
        return NO;
    }
    
    auto isFirst = llama_memory_seq_pos_max(llama_get_memory(self.context), 0) == -1;
    auto numPromptTokens = -llama_tokenize(self.vocab, prompt.UTF8String, (int32_t)prompt.length, nullptr, 0, isFirst, true);
    self->promptTokens.resize(numPromptTokens);
    if (llama_tokenize(self.vocab, prompt.UTF8String, (int32_t)prompt.length, self->promptTokens.data(), (int32_t)self->promptTokens.size(), isFirst, true) < 0) {
        NSLog(@"failed to tokenize prompt");
        return NO;
    }

    self.batch = llama_batch_get_one(self->promptTokens.data(), (int32_t)self->promptTokens.size());
    
    auto contextSize = llama_n_ctx(self.context);
    auto contextUsed = llama_memory_seq_pos_max(llama_get_memory(self.context), 0) + 1;
    if (contextUsed + batch.n_tokens > contextSize) {
        NSLog(@"context size exceeded: %d tokens used, %d tokens available", contextUsed + batch.n_tokens, contextSize);
        return NO;
    }
    
    auto ret = llama_decode(self.context, self.batch);
    if (ret != 0) {
        NSLog(@"failed to decode batch: %d", ret);
        return NO;
    }
    self->numCurrentTokens = self.batch.n_tokens;
    return YES;
}

/// Gets the next inferred response from the model
///
/// Here we retrieve a token from the sampler, and transform it into a
/// "piece" – literally the string representation of the token. Should
/// anything go wrong, or we reach the end of generation, we return an
/// empty string. Once the piece has been retrieved, we put the next token
/// into another `llama_batch` and decode it ready for the next iteration.
///
/// @return A `NSString` containing the next piece of the response,
/// or an empty string.
- (NSString *) getNextCompletion {
    self->nextTokenId = llama_sampler_sample(self.sampler, self.context, self.batch.n_tokens - 1);
    if (llama_vocab_is_eog(self.vocab, nextTokenId) || self->numCurrentTokens == 1024) {
        return @"";
    }
    
    char buf[256];
    auto pieceSize = llama_token_to_piece(self.vocab, nextTokenId, buf, sizeof(buf) - 1, 0, true);
    if (pieceSize < 0) {
        NSLog(@"failed to convert token to piece");
        return @"";
    }
    buf[pieceSize] = '\0';
    NSString *piece = [NSString stringWithUTF8String:buf];
    
    self.batch = llama_batch_get_one(&self->nextTokenId, 1);
    auto ret = llama_decode(self.context, self.batch);
    if (ret != 0) {
        NSLog(@"failed to decode batch: %d", ret);
        return @"";
    }
    self->numCurrentTokens += self.batch.n_tokens;
    return piece;
}

/// Convenience method to respond to a chat-like prompt
///
/// We process the prompt using the chat template (if requested), and
/// iterate over the model completion to obtain the full response. This
/// method will block the calling thread for a period of time, so it should
/// be called from a non-UI thread.
///
/// @param prompt The prompt to respond to.
/// @param useTemplate A `BOOL` indicating whether to use the chat template.
/// @return A `NSString` containing the response to the prompt, or an empty string.
- (NSString *) respondToPrompt:(NSString *)prompt usingTemplate:(BOOL)useTemplate {
    NSMutableString *response = [NSMutableString string];
    auto processedPrompt = prompt;
    if (useTemplate) {
        processedPrompt = [self processTemplateUsingPrompt:prompt];
    }
    
    if (![self initialiseCompletionForPrompt:processedPrompt]) {
        NSLog(@"failed to initialise completion for prompt: %@", prompt);
        return @"";
    }
    
    do {
        auto nextPiece = [self getNextCompletion];
        if (nextPiece.length == 0) {
            break;
        }
        [response appendString:nextPiece];
    } while(true);
    
    if (!self->messages.empty()) {
        self->messages.push_back({LLAMA_CHAT_ASSISTANT, strdup(response.UTF8String)});
        [self finaliseUsingTemplate:response];
    }
    
    return response;
}

/// Pre-process the prompt using the chat template (if available)
///
/// This method stores the adds the prompt to the list of
/// `llama_chat_message`s. It then renders the chat template into buffer
/// using the list of messages (expanding the buffer allocation as necessary),
/// and copies the fully rendered template into a new `NSString`.
///
/// @param prompt The prompt to process.
/// @return A `NSString` containing the processed template, or the original
/// prompt if the template could not be used.
- (NSString *) processTemplateUsingPrompt:(NSString *)prompt {
    if (!self->tmpl || strlen(self->tmpl) == 0) {
        NSLog(@"failed to get chat template from model.");
        return prompt;
    }

    self->messages.push_back({LLAMA_CHAT_USER, strdup([prompt UTF8String])});
    auto contextSize = llama_n_ctx(self.context);
    NSMutableData *formattedTemplate = [[NSMutableData alloc] initWithLength:contextSize];

    auto len = llama_chat_apply_template(tmpl, self->messages.data(), self->messages.size(), true, (char *)[formattedTemplate mutableBytes], (int32_t)contextSize);
    if (len > contextSize) {
        [formattedTemplate increaseLengthBy:len - contextSize + 1];
        len = llama_chat_apply_template(tmpl, self->messages.data(), self->messages.size(), true, (char *)[formattedTemplate mutableBytes], len);
    }
    if (len < 0) {
        NSLog(@"failed to apply chat template");
        for (auto &msg : self->messages) {
            free(const_cast<char *>(msg.content));
        }
        return prompt;
    }
    
    return [[NSString alloc] initWithBytes:[formattedTemplate bytes] length:len encoding:NSUTF8StringEncoding];
}

/// Finalises the response using the chat template.
///
/// This method simply checks that the template can be with the generated
/// response.
///
/// @param response The response to finalise using the template.
- (void) finaliseUsingTemplate:(NSString *)response {
    if (!self->tmpl || strlen(self->tmpl) == 0) {
        NSLog(@"no template to finalise response with");
        return;
    }
    
    auto len = llama_chat_apply_template(self->tmpl, self->messages.data(), self->messages.size(), false, nullptr, 0);
    if (len < 0) {
        NSLog(@"failed to finalise response using template");
    }
}

- (void) dealloc {
    for (auto &msg : self->messages) {
        free(const_cast<char *>(msg.content));
    }
    if (sampler) {
        llama_sampler_free(sampler);
        sampler = nullptr;
    }
    if (context) {
        llama_free(context);
        context = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
}
@end
