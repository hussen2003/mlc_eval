import * as webllm from "https://esm.run/@mlc-ai/web-llm";

// 1. Define the 3 pairs of models
const customAppConfig = {
  model_list: [
    // --- Llama 3.1 Pair ---
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC",
      model_id: "Llama-3.1-8B-Instruct-q4f16_1-MLC",
      model_lib: webllm.modelLibURLPrefix + webllm.modelVersion + "/Llama-3_1-8B-Instruct-q4f16_1_cs1k-webgpu.wasm",
      vram_required_MB: 5001.0,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: { context_window_size: 4096 },
    },
    {
      model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q3f16_1-MLC",
      model_id: "Llama-3.1-8B-Instruct-q3f16_1-MLC",
      model_lib: "http://localhost:8000/Llama-3_1-8B-Instruct-q3f16_1_cs1k-webgpu.wasm", 
      vram_required_MB: 3800.0,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: { context_window_size: 4096 },
    },

    // --- Gemma 2 Pair ---
    {
      model: "https://huggingface.co/mlc-ai/gemma-2-9b-it-q4f16_1-MLC",
      model_id: "gemma-2-9b-it-q4f16_1-MLC",
      model_lib: webllm.modelLibURLPrefix + webllm.modelVersion + "/gemma-2-9b-it-q4f16_1_cs1k-webgpu.wasm",
      vram_required_MB: 6422.01,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: { context_window_size: 4096 },
    },
    {
      model: "https://huggingface.co/mlc-ai/gemma-2-9b-it-q3f16_1-MLC",
      model_id: "gemma-2-9b-it-q3f16_1-MLC",
      model_lib: "http://localhost:8000/gemma-2-9b-it-q3f16_1_cs1k-webgpu.wasm", 
      vram_required_MB: 4800.0,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: { context_window_size: 4096 },
    },

    // --- Mistral v0.2 Pair ---
    {
      model: "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC",
      model_id: "Mistral-7B-Instruct-v0.2-q4f16_1-MLC",
      model_lib: webllm.modelLibURLPrefix + webllm.modelVersion + "/Mistral-7B-Instruct-v0.3-q4f16_1_cs1k-webgpu.wasm",
      vram_required_MB: 4573.39,
      low_resource_required: false,
      required_features: ["shader-f16"],
      overrides: { context_window_size: 4096 },
    },
    {
      model: "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q3f16_1-MLC",
      model_id: "Mistral-7B-Instruct-v0.2-q3f16_1-MLC",
      model_lib: "http://localhost:8000/Mistral-7B-Instruct-v0.2-q3f16_1_cs1k-webgpu.wasm", 
      vram_required_MB: 3500.0,
      low_resource_required: true,
      required_features: ["shader-f16"],
      overrides: { context_window_size: 4096 },
    }
  ]
};

let messages = [
  { content: "You are a helpful AI assistant.", role: "system" }
];
let currentPrompt = "";

const availableModels = customAppConfig.model_list.map((m) => m.model_id);
let selectedModel = availableModels[0];

function updateEngineInitProgressCallback(report) {
  console.log("initialize", report.progress);
  document.getElementById("download-status").textContent = report.text;
}

const engineConfig = { appConfig: customAppConfig };
const engine = new webllm.MLCEngine(engineConfig);
engine.setInitProgressCallback(updateEngineInitProgressCallback);

async function initializeWebLLMEngine() {
  document.getElementById("download-status").classList.remove("hidden");
  selectedModel = document.getElementById("model-selection").value;

  messages = [{ content: "You are a helpful AI assistant.", role: "system" }];
  document.getElementById("chat-box").innerHTML = ""; 
  const config = { temperature: 0.1, top_p: 1 }; 

  document.getElementById("send").disabled = true;
  await engine.reload(selectedModel, config);
  document.getElementById("send").disabled = false;
}

async function streamingGenerating(messages, onUpdate, onFinish, onError) {
  try {
    let curMessage = "";
    let isFirstToken = true;
    let ttftSeconds = 0;
    
    const startTime = performance.now();

    const completion = await engine.chat.completions.create({
      stream: true,
      messages,
    });

    for await (const chunk of completion) {
      if (isFirstToken) {
        const firstTokenTime = performance.now();
        ttftSeconds = (firstTokenTime - startTime) / 1000;
        isFirstToken = false;
      }

      const curDelta = chunk.choices[0].delta.content;
      if (curDelta) { curMessage += curDelta; }
      onUpdate(curMessage);
    }
    
    const finalMessage = await engine.getMessage();
    const stats = await engine.runtimeStatsText();
    onFinish(finalMessage, ttftSeconds, stats);
  } catch (err) {
    onError(err);
  }
}

function onMessageSend() {
  const input = document.getElementById("user-input").value.trim();
  if (input.length === 0) return;

  currentPrompt = input;
  const message = { content: input, role: "user" };
  document.getElementById("send").disabled = true;

  messages.push(message);
  appendMessage(message);

  document.getElementById("user-input").value = "";
  document.getElementById("user-input").setAttribute("placeholder", "Generating...");

  const aiMessage = { content: "typing...", role: "assistant" };
  appendMessage(aiMessage);

  const onFinishGenerating = (finalMessage, ttft, statsText) => {
    updateLastMessage(finalMessage);
    document.getElementById("send").disabled = false;
    document.getElementById("user-input").setAttribute("placeholder", "Type your benchmark prompt...");
    
    document.getElementById('chat-stats').classList.remove('hidden');
    document.getElementById('chat-stats').textContent = statsText + ` | TTFT: ${ttft.toFixed(2)}s`;

    const match = statsText.match(/decoding:\s*([0-9.]+)\s*tokens\/sec/);
    const tps = match ? match[1] : "N/A";

    appendBenchmarkRow(selectedModel, currentPrompt, finalMessage, ttft.toFixed(3) + "s", tps);
  };

  streamingGenerating(messages, updateLastMessage, onFinishGenerating, console.error);
}

function appendBenchmarkRow(model, prompt, response, ttft, tps) {
  const tbody = document.querySelector("#benchmark-table tbody");
  const row = document.createElement("tr");
  row.innerHTML = `
    <td><strong>${model}</strong></td>
    <td>${prompt}</td>
    <td>${response}</td>
    <td>${ttft}</td>
    <td>${tps}</td>
  `;
  tbody.appendChild(row);
}

function appendMessage(message) {
  const chatBox = document.getElementById("chat-box");
  const container = document.createElement("div");
  container.classList.add("message-container", message.role);

  const newMessage = document.createElement("div");
  newMessage.classList.add("message");
  newMessage.textContent = message.content;

  container.appendChild(newMessage);
  chatBox.appendChild(container);
  chatBox.scrollTop = chatBox.scrollHeight; 
}

function updateLastMessage(content) {
  const messageDoms = document.getElementById("chat-box").querySelectorAll(".message");
  if (messageDoms.length > 0) {
    const lastMessageDom = messageDoms[messageDoms.length - 1];
    lastMessageDom.textContent = content;
  }
}

const selectElement = document.getElementById("model-selection");
availableModels.forEach((modelId) => {
  const option = document.createElement("option");
  option.value = modelId;
  option.textContent = modelId;
  selectElement.appendChild(option);
});

selectElement.value = selectedModel;

document.getElementById("download").addEventListener("click", initializeWebLLMEngine);
document.getElementById("send").addEventListener("click", onMessageSend);

document.getElementById("user-input").addEventListener("keydown", function(event) {
  if (event.key === "Enter") {
    event.preventDefault(); 
    if (!document.getElementById("send").disabled) {
      onMessageSend();
    }
  }
});