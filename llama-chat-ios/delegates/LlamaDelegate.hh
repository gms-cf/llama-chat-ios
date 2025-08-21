//
//  LlamaDelegate.hh
//  llama-chat-objc
//
//  Created by George MacKay-Shore on 19/08/2025.
//

#ifndef LLAMA_DELEGATE_H
#define LLAMA_DELEGATE_H
#import <Foundation/Foundation.h>
#include <llama/llama.h>

#define LLAMA_CHAT_USER "user"
#define LLAMA_CHAT_ASSISTANT "assistant"
#define LLAMA_CHAT_SYSTEM "system"

@interface LlamaDelegate : NSObject
@property (nonatomic) struct llama_model *model;
@property (nonatomic) struct llama_context *context;
@property (nonatomic) const struct llama_vocab *vocab;
@property (nonatomic) struct llama_sampler *sampler;
@property (nonatomic) struct llama_batch batch;

+ (instancetype) newDelegateWithModelName:(NSString *)modelName;

- (instancetype) initWithModel:(struct llama_model *)model andContext:(struct llama_context *)context;
- (BOOL) initialiseCompletionForPrompt:(NSString *)prompt;
- (NSString *) getNextCompletion;
- (NSString *) respondToPrompt:(NSString *)prompt usingTemplate:(BOOL)useTemplate;
@end
#endif // LLAMA_DELEGATE_H
