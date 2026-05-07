import SwiftUI

struct TextBox: View {
    let fullText: String
    @State private var displayedText = ""
    
    var body: some View {
        Text(displayedText)
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.systemGray6))
            .cornerRadius(10)
            .onChange(of: fullText) {
                animateText()
            }
    }
    
    func animateText() {
        displayedText = ""
        for (index, letter) in fullText.enumerated() {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(index) * 0.03) {
                displayedText.append(letter)
            }
        }
    }
}

struct ContentView: View {
    @State private var text = ""
    @State private var response = ""
    @State private var loading = false

    let mlcURL = "http://100.x.x.x:8000" // replace with your Tailscale IP

    var body: some View {
        VStack(spacing: 16) {
            Spacer()

            if !response.isEmpty {
                TextBox(fullText: response)
            }

            if loading {
                ProgressView()
            }

            TextField("Enter text here", text: $text)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.blue, lineWidth: 1)
                )

            Button("Go") {
                sendMessage()
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(loading ? Color.gray : Color.blue)
            .foregroundStyle(.white)
            .cornerRadius(10)
            .font(.headline)
            .disabled(loading)
        }
        .padding()
    }

    func sendMessage() {
        guard !text.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        loading = true
        response = ""
        let mlcURL = "http://100.123.84.64:8000"

        let url = URL(string: "\(mlcURL)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "model": "Llama-3.2-1B-Instruct-q4f16_1-MLC",
            "messages": [["role": "user", "content": text]]
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        URLSession.shared.dataTask(with: request) { data, _, error in
            DispatchQueue.main.async {
                loading = false
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let choices = json["choices"] as? [[String: Any]],
                   let message = choices.first?["message"] as? [String: Any],
                   let content = message["content"] as? String {
                    response = content
                } else {
                    response = "Error: could not reach server"
                }
            }
        }.resume()
    }
}

#Preview {
    ContentView()
}
