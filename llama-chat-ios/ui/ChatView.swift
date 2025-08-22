//
//  ChatView.swift
//  llama-chat-ios
//
//  Created by George MacKay-Shore on 22/08/2025.
//

import SwiftUI

extension LlamaDelegate: @unchecked Sendable {}

struct ChatView: View {
    private var llamaDelegate: LlamaDelegate?
    @State private var multiLineText: String = ""
    @State private var output: String = "Welcome to Llama Chat!\n"
    @FocusState private var textFieldIsFocused: Bool
    
    init(_ initialiseDelegate: Bool = false) {
        if initialiseDelegate {
            self.llamaDelegate = LlamaDelegate.newDelegate(withModelName: "model")
        }
    }
    
    func sendMessage() {
        textFieldIsFocused = false
        let userMessage = multiLineText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !userMessage.isEmpty else { return }
        output += "\n**You**: \(userMessage)"
        multiLineText = ""
        guard let llamaDelegate else { return }

        // Because LlamaDelegate is an ObjC class, we dispatch to a global
        // queue to avoid blocking the main thread.
        DispatchQueue.global(qos: .userInitiated).async {
            let response = llamaDelegate.respond(toPrompt: userMessage, usingTemplate: true)!
            DispatchQueue.main.async {
                output += "\n\(response)\n"
            }
        }
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Chat messages area
            ScrollViewReader { proxy in
                Text("Llama Chat")
                    .font(.largeTitle)
                    .padding(.vertical, 10)
                    .frame(maxWidth: .infinity, alignment: .center)
                
                ScrollView {
                    VStack {
                        Text(.init(output))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal)
                            .padding(.top, 10)
                        
                        // Invisible view at the bottom for scrolling
                        Color.clear
                            .frame(height: 1)
                            .id("bottomID")
                    }
                }
                .onChange(of: output) {
                    withAnimation {
                        proxy.scrollTo("bottomID", anchor: .bottom)
                    }
                }
                .onAppear {
                    // Scroll to bottom when view appears
                    proxy.scrollTo("bottomID", anchor: .bottom)
                }
            }
            .frame(maxWidth: .infinity)
            .background(Color(.systemBackground))
            
            Divider()
            
            // Input area
            HStack(alignment: .bottom) {
                // Text editor
                ZStack(alignment: .leading) {
                    TextEditor(text: $multiLineText)
                        .padding(4)
                        .cornerRadius(10)
                        .frame(minHeight: 40, maxHeight: 120)
                        .focused($textFieldIsFocused)
                }
                
                // Send button
                Button(action: sendMessage) {
                    Image(systemName: "arrow.up.circle.fill")
                        .resizable()
                        .frame(width: 32, height: 32)
                        .foregroundColor(.blue)
                }
            }
            .padding()
        }
    }
}

#Preview {
    ChatView()
}
