//
//  ChatViewWrapper.swift
//  llama-chat-ios
//
//  Created by George MacKay-Shore on 22/08/2025.
//

import SwiftUI
import UIKit

public class ChatViewFactory: NSObject {
    @MainActor @objc static func createChatView() -> UIViewController {
        return UIHostingController(rootView: ChatView(true))
    }
}
