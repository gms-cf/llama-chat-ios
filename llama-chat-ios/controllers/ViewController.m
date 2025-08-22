//
//  ViewController.m
//  llama-chat-ios
//
//  Created by George MacKay-Shore on 20/08/2025.
//

#import "ViewController.h"
#import "llama_chat_ios-Swift.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    UIViewController *chatViewController = [ChatViewFactory createChatView];
    
    [self addChildViewController:chatViewController];
    [self.view addSubview:chatViewController.view];
    
    chatViewController.view.frame = self.view.bounds;
    [chatViewController didMoveToParentViewController:self];
}


@end
