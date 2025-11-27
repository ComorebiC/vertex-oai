import { Buffer } from "node:buffer";

// --- 全局常量更新为 Vertex AI Express Mode 的地址 ---
const TARGET_BASE_URL = "https://aiplatform.googleapis.com";
const TARGET_PREFIX = "/v1/publishers/google/models";

// 定义图片生成模型列表，用于特殊处理配置
const IMAGE_MODELS = [
  "gemini-3-pro-image-preview",
  "gemini-2.5-flash-image",
  "gemini-2.0-flash-exp-image"
];

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const auth = request.headers.get("Authorization");
    const apiKey = auth?.split(" ")[1];

    const errHandler = (err) => {
      console.error(err);
      return new Response(err.message, fixCors({ status: err.status ?? 500 }));
    };
    try {
      if (!apiKey) {
        throw new HttpError("Missing API Key (Authorization header)", 401);
      }

      const assert = (success) => {
        if (!success) {
          throw new HttpError("The specified HTTP method is not allowed for the requested resource", 400);
        }
      };
      const { pathname } = new URL(request.url);
      switch (true) {
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST");
          return handleCompletions(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/audio/speech"):
          assert(request.method === "POST");
          return handleSpeech(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST");
          return handleEmbeddings(await request.json(), apiKey)
            .catch(errHandler);
        case pathname.endsWith("/models"):
          assert(request.method === "GET");
          return handleModels(apiKey)
            .catch(errHandler);
        default:
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  }
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  return { headers, status, statusText };
};


const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
    }
  });
};

const API_CLIENT = "genai-js/0.21.0";
const makeHeaders = (more) => ({
  "x-goog-api-client": API_CLIENT,
  ...more
});

async function handleModels(apiKey) {
  const url = `${TARGET_BASE_URL}${TARGET_PREFIX}`;
  const response = await fetch(url, {
    headers: makeHeaders(),
  });
  let { body } = response;
  if (response.ok) {
    const { models } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: (models || []).map(({ name }) => ({
        id: name.replace("models/", ""),
        object: "model",
        created: 0,
        owned_by: "",
      })),
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_EMBEDDINGS_MODEL = "gemini-embedding-001";
async function handleEmbeddings(req, apiKey) {
  let modelFull, model;
  switch (true) {
    case typeof req.model !== "string":
      throw new HttpError("model is not specified", 400);
    case req.model.startsWith("models/"):
      modelFull = req.model;
      model = modelFull.substring(7);
      break;
    case req.model.startsWith("gemini-"):
      model = req.model;
      break;
    default:
      model = DEFAULT_EMBEDDINGS_MODEL;
  }
  modelFull = modelFull ?? "models/" + model;

  if (!Array.isArray(req.input)) {
    req.input = [req.input];
  }

  const url = `${TARGET_BASE_URL}${TARGET_PREFIX}/${model}:batchEmbedContents`;
  const newURL = new URL(url);
  newURL.searchParams.set("key", apiKey);

  const response = await fetch(newURL.toString(), {
    method: "POST",
    headers: makeHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      "requests": req.input.map(text => ({
        model: modelFull,
        content: { parts: [{ text }] },
        outputDimensionality: req.dimensions,
      }))
    })
  });
  let { body } = response;
  if (response.ok) {
    const { embeddings } = JSON.parse(await response.text());
    body = JSON.stringify({
      object: "list",
      data: embeddings.map(({ values }, index) => ({
        object: "embedding",
        index,
        embedding: values,
      })),
      model,
    }, null, "  ");
  }
  return new Response(body, fixCors(response));
}

function addWavHeader(pcmData) {
  const sampleRate = 24000;
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = pcmData.length;
  const chunkSize = 36 + dataSize;
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(chunkSize, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(numChannels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36);
  header.writeUInt32LE(dataSize, 40);
  return Buffer.concat([header, pcmData]);
}

async function handleTts(req, apiKey) {
  if (!req.messages || req.messages.length === 0) {
    throw new HttpError("`messages` array is required for TTS.", 400);
  }
  if (!req.audio?.voice) {
    throw new HttpError("`audio.voice` is required for TTS.", 400);
  }
  const lastMessage = req.messages[req.messages.length - 1];
  const parts = await transformMsg(lastMessage);
  const inputText = parts.map(p => p.text).join(' ');
  if (!inputText) {
    throw new HttpError("A non-empty text message is required for TTS.", 400);
  }

  const geminiTtsModel = req.model || "gemini-2.5-flash-preview-tts";

  const geminiPayload = {
    model: geminiTtsModel,
    contents: [{
      parts: [{ text: inputText }]
    }],
    generationConfig: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: {
            voiceName: req.audio.voice
          }
        }
      }
    },
  };

  const url = `${TARGET_BASE_URL}${TARGET_PREFIX}/${geminiTtsModel}:generateContent`;
  const newURL = new URL(url);
  newURL.searchParams.set("key", apiKey);

  const response = await fetch(newURL.toString(), {
    method: "POST",
    headers: makeHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(geminiPayload),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    console.error("Gemini TTS API Error:", errorBody);
    return new Response(errorBody, fixCors(response));
  }
  const geminiResponse = await response.json();
  const audioDataBase64 = geminiResponse.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!audioDataBase64) {
    console.error("Could not extract audio data from Gemini response:", JSON.stringify(geminiResponse));
    throw new HttpError("Failed to generate audio, invalid response from upstream.", 500);
  }
  const requestedFormat = req.audio.format || 'wav';
  let finalAudioDataB64 = audioDataBase64;
  let finalFormat = 'pcm_s16le_24000_mono';
  if (requestedFormat.toLowerCase() === 'wav') {
    const pcmData = Buffer.from(audioDataBase64, 'base64');
    const wavData = addWavHeader(pcmData);
    finalAudioDataB64 = wavData.toString('base64');
    finalFormat = 'wav';
  }
  const openAiResponse = {
    id: "chatcmpl-tts-" + generateId(),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: req.model,
    choices: [{
      index: 0,
      message: {
        role: "assistant",
        audio: {
          format: finalFormat,
          data: finalAudioDataB64,
          transcript: inputText,
        }
      },
      finish_reason: "stop",
    }],
    usage: null,
  };
  return new Response(JSON.stringify(openAiResponse, null, 2), fixCors({ headers: response.headers }));
}

async function handleSpeech(req, apiKey) {
  if (!req.input) {
    throw new HttpError("`input` field is required.", 400);
  }
  if (!req.voice) {
    throw new HttpError("`voice` field is required.", 400);
  }
  const geminiTtsModel = req.model || "gemini-2.5-flash-preview-tts";
  const geminiPayload = {
    model: geminiTtsModel,
    contents: [{
      parts: [{ text: req.input }]
    }],
    generationConfig: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: {
            voiceName: req.voice
          }
        }
      }
    },
  };

  const url = `${TARGET_BASE_URL}${TARGET_PREFIX}/${geminiTtsModel}:generateContent`;
  const newURL = new URL(url);
  newURL.searchParams.set("key", apiKey);

  const geminiApiResponse = await fetch(newURL.toString(), {
    method: "POST",
    headers: makeHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(geminiPayload),
  });

  if (!geminiApiResponse.ok) {
    const errorBody = await geminiApiResponse.text();
    console.error("Gemini TTS API Error:", errorBody);
    return new Response(errorBody, fixCors({ headers: geminiApiResponse.headers, status: geminiApiResponse.status, statusText: geminiApiResponse.statusText }));
  }
  const geminiResponseJson = await geminiApiResponse.json();
  const audioDataBase64 = geminiResponseJson.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!audioDataBase64) {
    throw new HttpError("Failed to extract audio data from Gemini response.", 500);
  }
  const pcmData = Buffer.from(audioDataBase64, 'base64');
  const responseFormat = req.response_format || 'wav';
  let audioData;
  let contentType;
  const corsHeaders = fixCors({}).headers;
  switch (responseFormat.toLowerCase()) {
    case 'wav':
      audioData = addWavHeader(pcmData);
      contentType = 'audio/wav';
      break;
    case 'pcm':
      audioData = pcmData;
      contentType = 'audio/L16; rate=24000; channels=1';
      break;
    case 'mp3':
    case 'opus':
    case 'aac':
    case 'flac':
    default:
      audioData = addWavHeader(pcmData);
      contentType = 'audio/wav';
      corsHeaders.set('X-Warning', `Unsupported format "${responseFormat}" requested, fallback to "wav".`);
      break;
  }
  corsHeaders.set('Content-Type', contentType);
  return new Response(audioData, {
    status: 200,
    headers: corsHeaders
  });
}


const DEFAULT_MODEL = "gemini-2.5-flash";
async function handleCompletions(req, apiKey) {
  const isTtsRequest = Array.isArray(req.modalities) && req.modalities.includes("audio");
  if (isTtsRequest) {
    return handleTts(req, apiKey);
  }
  let model;
  switch (true) {
    case typeof req.model !== "string":
      break;
    case req.model.startsWith("models/"):
      model = req.model.substring(7);
      break;
    case req.model.startsWith("gemini-"):
    case req.model.startsWith("gemma-"):
    case req.model.startsWith("learnlm-"):
      model = req.model;
  }
  model = model || DEFAULT_MODEL;

  let targetModel = model;
  if (model.includes("gemini-2.5-pro-actually-3")) {
    console.log(`[Proxy] Redirecting model: ${model} -> gemini-3-pro-preview`);
    targetModel = "gemini-3-pro-preview";
  }

  let body = await transformRequest(req, targetModel);

  if (IMAGE_MODELS.includes(targetModel)) {
    body.generationConfig = body.generationConfig || {};
    if (!body.generationConfig.responseModalities) {
        body.generationConfig.responseModalities = ["TEXT", "IMAGE"];
    }
    delete body.system_instruction;
  }

  const extra = req.extra_body?.google;
  if (extra) {
    if (extra.safety_settings) {
      body.safetySettings = extra.safety_settings;
    }
    if (extra.cached_content) {
      body.cachedContent = extra.cached_content;
    }
    if (extra.thinking_config) {
      body.generationConfig.thinkingConfig = extra.thinking_config;
    }
    if (extra.image_config) {
      body.generationConfig.imageConfig = extra.image_config;
    }
  }

  switch (true) {
    case targetModel.endsWith(":search"):
      targetModel = targetModel.slice(0, -7);
      body.tools = body.tools || [];
      body.tools.push({ "googleSearch": {} });
      break;
    case targetModel.endsWith(":url"):
      targetModel = targetModel.slice(0, -4);
      body.tools = body.tools || [];
      body.tools.push({ "url_context": {} });
      break;
    case targetModel.endsWith(":execode"):
      targetModel = targetModel.slice(0, -8);
      body.tools = body.tools || [];
      body.tools.push({ "code_execution": {} });
      break;
  }

  const TASK = req.stream ? "streamGenerateContent" : "generateContent";

  let url = `${TARGET_BASE_URL}${TARGET_PREFIX}/${targetModel}:${TASK}`;
  const newURL = new URL(url);
  newURL.searchParams.set("key", apiKey);
  if (req.stream) {
    newURL.searchParams.set("alt", "sse");
  }
  url = newURL.toString();

  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  body = response.body;
  if (response.ok) {
    let id = "chatcmpl-" + generateId();
    const shared = {};
    if (req.stream) {
      body = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(new TransformStream({
          transform: parseStream,
          flush: parseStreamFlush,
          buffer: "",
          shared,
        }))
        .pipeThrough(new TransformStream({
          transform: toOpenAiStream,
          flush: toOpenAiStreamFlush,
          streamIncludeUsage: req.stream_options?.include_usage,
          model: targetModel,
          id, last: [],
          shared,
        }))
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      try {
        body = JSON.parse(body);
        if (!body.candidates) {
          throw new Error("Invalid completion object");
        }
      } catch (err) {
        console.error("Error parsing response:", err);
        return new Response(body, fixCors(response));
      }
      body = processCompletionsResponse(body, targetModel, id);
    }
  }
  return new Response(body, fixCors(response));
}


const resolveRef = (ref, rootSchema) => {
  if (!ref.startsWith('#/')) {
    return null;
  }
  const path = ref.substring(2).split('/');
  let current = rootSchema;
  for (const segment of path) {
    if (current && typeof current === 'object' && segment in current) {
      current = current[segment];
    } else {
      return null;
    }
  }
  return current;
};

const transformOpenApiSchemaToGemini = (schemaNode, rootSchema, visited = new Set()) => {
  if (typeof schemaNode !== "object" || schemaNode === null || visited.has(schemaNode)) {
    return;
  }
  visited.add(schemaNode);

  if (schemaNode.$ref) {
    const resolved = resolveRef(schemaNode.$ref, rootSchema);
    if (resolved) {
      delete schemaNode.$ref;
      Object.assign(schemaNode, { ...JSON.parse(JSON.stringify(resolved)), ...schemaNode });
    }
  }

  if (Array.isArray(schemaNode)) {
    schemaNode.forEach(item => transformOpenApiSchemaToGemini(item, rootSchema, visited));
    return;
  }

  if (schemaNode.type) {
    const typeMap = {
      "string": "STRING", "number": "NUMBER", "integer": "INTEGER",
      "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT",
    };
    const primaryType = Array.isArray(schemaNode.type)
      ? schemaNode.type.find(t => t !== "null")
      : schemaNode.type;

    if (primaryType && typeMap[primaryType.toLowerCase()]) {
      schemaNode.type = typeMap[primaryType.toLowerCase()];
    } else {
      delete schemaNode.type;
    }
  }

  if (schemaNode.type === 'ARRAY' && !schemaNode.items) {
    schemaNode.items = { type: 'OBJECT' };
  }

  if (Array.isArray(schemaNode.anyOf)) {
    schemaNode.anyOf.forEach(item => transformOpenApiSchemaToGemini(item, rootSchema, visited));

    if (schemaNode.anyOf.every(item => item && typeof item === 'object' && item.hasOwnProperty('const'))) {
      const enumValues = schemaNode.anyOf
        .map(item => item.const)
        .filter(val => val !== "" && val !== null)
        .map(String);
      if (enumValues.length > 0) {
        schemaNode.type = 'STRING';
        schemaNode.enum = enumValues;
      }
    } else if (!schemaNode.type) {
      const firstValidItem = schemaNode.anyOf.find(item => item && (item.type || item.enum));
      if (firstValidItem) {
        Object.assign(schemaNode, firstValidItem);
      }
    }
    delete schemaNode.anyOf;
  }

  const unsupportedKeys = [
    'title', '$schema', '$ref', 'strict', 'exclusiveMaximum',
    'exclusiveMinimum', 'additionalProperties', 'oneOf', 'allOf', 'default',
    '$defs'
  ];
  unsupportedKeys.forEach(key => delete schemaNode[key]);

  if (schemaNode.properties) {
    Object.values(schemaNode.properties).forEach(prop => transformOpenApiSchemaToGemini(prop, rootSchema, visited));
  }
  if (schemaNode.items) {
    transformOpenApiSchemaToGemini(schemaNode.items, rootSchema, visited);
  }
};

const adjustSchema = (tool) => {
  const parameters = tool.function?.parameters;
  if (parameters) {
    transformOpenApiSchemaToGemini(parameters, parameters);
  }
};

const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map(category => ({
  category,
  threshold: "BLOCK_NONE",
}));

const transformConfig = (req, model) => {
  let cfg = {};

  const fieldsMap = {
    frequency_penalty: "frequencyPenalty",
    max_completion_tokens: "maxOutputTokens",
    max_tokens: "maxOutputTokens",
    n: "candidateCount",
    presence_penalty: "presencePenalty",
    seed: "seed",
    stop: "stopSequences",
    temperature: "temperature",
    top_k: "topK",
    top_p: "topP",
  };

  for (let key in req) {
    const matchedKey = fieldsMap[key];
    if (matchedKey && req[key] !== null) {
      cfg[matchedKey] = req[key];
    }
  }

  if (req.response_format) {
    switch (req.response_format.type) {
      case "json_schema":
        if (req.response_format.json_schema?.schema) {
            adjustSchema(req.response_format);
            cfg.responseSchema = req.response_format.json_schema.schema;
            cfg.responseMimeType = "application/json";
        }
        break;
      case "json_object":
        cfg.responseMimeType = "application/json";
        break;
      case "text":
        cfg.responseMimeType = "text/plain";
        break;
    }
  }

  if (req.reasoning_effort) {
    const isV3 = model?.includes("gemini-3");

    if (isV3) {
      let thinkingLevel;
      switch (req.reasoning_effort) {
        case "low":
          thinkingLevel = "LOW";
          break;
        case "medium":
          thinkingLevel = "MEDIUM";
          break;
        case "high":
          thinkingLevel = "HIGH";
          break;
        default:
          thinkingLevel = "HIGH";
      }
      cfg.thinkingConfig = { thinkingLevel, includeThoughts: true };
    }
    else {
      let thinkingBudget;
      const isPro = model?.includes("pro");
      const isLite = model?.includes("lite");

      switch (req.reasoning_effort) {
        case "low":
          thinkingBudget = isLite ? 1024 : 2048;
          break;
        case "medium":
          thinkingBudget = -1;
          break;
        case "high":
          thinkingBudget = isPro ? 32768 : 24576;
          break;
        default:
          thinkingBudget = -1;
      }

      if (typeof thinkingBudget !== "undefined") {
        cfg.thinkingConfig = { thinkingBudget, includeThoughts: true };
      }
    }
  }

  return cfg;
};

const parseImg = async (url) => {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText} (${url})`);
      }
      mimeType = response.headers.get("content-type");
      data = Buffer.from(await response.arrayBuffer()).toString("base64");
    } catch (err) {
      throw new Error("Error fetching image: " + err.toString());
    }
  } else {
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match) {
      throw new HttpError("Invalid image data: " + url, 400);
    }
    ({ mimeType, data } = match.groups);
  }
  return {
    inlineData: {
      mimeType,
      data,
    },
  };
};

const transformFnResponse = ({ content, tool_call_id }, parts) => {
  if (!parts.calls) {
    throw new HttpError("No function calls found in the previous message", 400);
  }

  let stringContent;
  if (Array.isArray(content)) {
    const textPart = content.find(part => part.type === 'text');
    stringContent = textPart?.text;
  } else if (typeof content === 'string') {
    stringContent = content;
  }

  if (typeof stringContent !== 'string') {
    throw new HttpError("Could not extract a valid string from the tool response content.", 400);
  }

  let response;
  try {
    response = JSON.parse(stringContent);
  } catch (err) {
    response = { result: stringContent };
  }

  if (typeof response !== "object" || response === null || Array.isArray(response)) {
    response = { result: response };
  }

  if (!tool_call_id) {
    throw new HttpError("tool_call_id not specified", 400);
  }
  const { i, name } = parts.calls[tool_call_id] ?? {};
  if (!name) {
    throw new HttpError("Unknown tool_call_id: " + tool_call_id, 400);
  }
  if (parts[i]) {
    throw new HttpError("Duplicated tool_call_id: " + tool_call_id, 400);
  }
  parts[i] = {
    functionResponse: {
      name,
      response,
    }
  };
};

const transformFnCalls = (message) => {
  const { tool_calls } = message;
  const signature = message.thought_signature || 
                    message.extra_content?.google?.thought_signature ||
                    tool_calls?.[0]?.extra_content?.google?.thought_signature;

  const calls = {};
  const parts = tool_calls.map(({ function: { arguments: argstr, name }, id, type }, i) => {
    if (type !== "function") {
      throw new HttpError(`Unsupported tool_call type: "${type}"`, 400);
    }
    let args;
    try {
      args = JSON.parse(argstr);
    } catch (err) {
      console.error("Error parsing function arguments:", err);
      throw new HttpError("Invalid function arguments: " + argstr, 400);
    }
    calls[id] = { i, name };
    
    const part = {
      functionCall: {
        name,
        args,
      }
    };

    if (i === 0 && signature) {
        part.thoughtSignature = signature;
    }
    return part;
  });
  parts.calls = calls;
  return parts;
};

// 辅助函数：从文本中提取 Base64 图片 Markdown 并转换为 parts
// stripImages = true: 移除 Markdown 图片链接，并且不生成 inlineData
function parseAssistantContent(content, stripImages = false) {
  const parts = [];
  const imageMarkdownRegex = /!\[gemini-image-generation\]\(data:(image\/\w+);base64,([\w+/=-]+)\)/g;

  if (typeof content !== 'string') {
    return parts;
  }

  let lastIndex = 0;
  let match;

  while ((match = imageMarkdownRegex.exec(content)) !== null) {
    // 提取图片之前的文本
    if (match.index > lastIndex) {
      parts.push({ text: content.substring(lastIndex, match.index) });
    }

    // 如果不需要剥离图片，则转换数据
    if (!stripImages) {
        const mimeType = match[1];
        const data = match[2];
        parts.push({
          inlineData: {
            mimeType,
            data,
          },
        });
    }

    lastIndex = match.index + match[0].length;
  }

  // 提取剩余文本
  if (lastIndex < content.length) {
    parts.push({ text: content.substring(lastIndex) });
  }

  // 修正逻辑：只有在非剥离模式下，如果没匹配到任何东西才把原始内容当做文本。
  // 在剥离模式下，如果全部被剥离，parts 为空是预期的。
  if (parts.length === 0 && content && !stripImages) {
    parts.push({ text: content });
  }

  return parts;
}

const transformMsg = async ({ content }) => {
  const parts = [];
  if (!Array.isArray(content)) {
    if (typeof content === 'string') {
      parts.push(...parseAssistantContent(content, false));
    }
    return parts;
  }

  for (const item of content) {
    switch (item.type) {
      case "input_text":
      case "text":
        if (item.text) {
          parts.push(...parseAssistantContent(item.text, false));
        }
        break;
      case "image_url":
        const imageUrlObject = Array.isArray(item.image_url) ? item.image_url[0] : item.image_url;
        if (imageUrlObject && imageUrlObject.url) {
          parts.push(await parseImg(imageUrlObject.url));
        }
        break;
      case "input_file": {
        let fileDataUri = item.file_data;
        if (!fileDataUri.startsWith("data:")) {
          fileDataUri = `data:application/pdf;base64,${item.file_data}`;
        }
        const match = fileDataUri.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
        if (!match) {
          throw new HttpError(`Invalid file_data format.`, 400);
        }
        const { mimeType, data } = match.groups;
        parts.push({ inlineData: { mimeType, data } });
        break;
      }
      case "file": {
        let fileDataUri = item.file.file_data;
        if (!fileDataUri.startsWith("data:")) {
          fileDataUri = `data:application/pdf;base64,${item.file_data}`;
        }
        const match = fileDataUri.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
        if (!match) {
          throw new HttpError(`Invalid file_data format.`, 400);
        }
        const { mimeType, data } = match.groups;
        parts.push({ inlineData: { mimeType, data } });
        break;
      }
      case "input_audio":
        parts.push({
          inlineData: {
            mimeType: "audio/" + item.input_audio.format,
            data: item.input_audio.data,
          }
        });
        break;
      default:
        throw new HttpError(`Unknown "content" item type: "${item.type}"`, 400);
    }
  }
  if (content.every(item => item.type === "image_url")) {
    parts.push({ text: "" });
  }
  return parts;
};

const transformMessages = async (messages, model) => {
  if (!messages) { return []; }
  const contents = [];
  let system_instruction;

  // 判断是否为 V3+ 图片模型 (Gemini 3 Pro Image)
  const isV3ImageModel = model && model.includes("gemini-3") && model.includes("image");

  for (const item of messages) {
    switch (item.role) {
      case "system":
        system_instruction = { parts: await transformMsg(item) };
        continue;
      case "tool":
        let { role: r, parts: p } = contents[contents.length - 1] ?? {};
        if (r !== "function") {
          const calls = p?.calls;
          p = []; p.calls = calls;
          contents.push({ role: "function", parts: p });
        }
        transformFnResponse(item, p);
        continue;
      case "assistant":
        item.role = "model";
        let modelParts = [];
        
        // 1. 处理 reasoning_content (作为 thought)
        if (item.reasoning_content) {
            modelParts.push({ text: item.reasoning_content, thought: true });
        }

        if (item.tool_calls) {
            const toolParts = transformFnCalls(item);
            modelParts.push(...toolParts);
        } else {
            const signature = item.thought_signature || item.extra_content?.google?.thought_signature;
            
            if (isV3ImageModel && signature) {
               // 【V3 逻辑】：
               let rawContent = typeof item.content === 'string' ? item.content : "";
               if (Array.isArray(item.content)) {
                   rawContent = item.content.map(c => c.text || "").join("");
               }
               
               // 剥离图片，不生成 inlineData
               const textParts = parseAssistantContent(rawContent, true); 
               
               // 【关键修复】严格过滤空文本 Part
               // 只有当 text 存在且不为空字符串时才保留
               const validTextParts = textParts.filter(p => p.text !== undefined && p.text !== "");

               modelParts.push(...validTextParts);
               modelParts.push({ thoughtSignature: signature });

            } else {
               // 【V2.5 逻辑】：正常保留 Base64 图片
               const contentParts = await transformMsg(item);
               modelParts.push(...contentParts);
            }
        }

        contents.push({
          role: item.role,
          parts: modelParts,
        });
        continue;
      case "user":
        contents.push({
          role: item.role,
          parts: item.tool_calls ? transformFnCalls(item) : await transformMsg(item)
        });
        break;
      default:
        throw new HttpError(`Unknown message role: "${item.role}"`, 400);
    }
  }
  if (system_instruction) {
    if (!contents[0]?.parts.some(part => part.text)) {
      contents.unshift({ role: "user", parts: [{ text: " " }] });
    }
  }
  return { system_instruction, contents };
};

const reverseTransformValue = (value) => {
  if (typeof value !== 'string') {
    return value;
  }
  if (value === 'true') return true;
  if (value === 'false') return false;
  if (value === 'null') return null;
  if (value.trim() !== '' && !isNaN(Number(value)) && Number(value).toString() === value) {
    return Number(value);
  }
  return value;
};

const reverseTransformArgs = (args) => {
  if (typeof args !== 'object' || args === null) {
    return args;
  }

  if (Array.isArray(args)) {
    return args.map(item => reverseTransformArgs(item));
  }

  const newArgs = {};
  for (const key in args) {
    if (Object.prototype.hasOwnProperty.call(args, key)) {
      const value = args[key];
      if (typeof value === 'object') {
        newArgs[key] = reverseTransformArgs(value);
      } else {
        newArgs[key] = reverseTransformValue(value);
      }
    }
  }
  return newArgs;
};

const transformTools = (req) => {
  let tools, toolConfig;

  if (req.tools) {
    const funcs = req.tools.filter(tool => tool.type === "function");

    funcs.forEach(adjustSchema);

    tools = [{
      functionDeclarations: funcs.map(schema => schema.function)
    }];
  }

  if (req.tool_choice) {
    let mode = "AUTO";
    let allowedFunctionNames;

    if (typeof req.tool_choice === "string") {
      switch (req.tool_choice) {
        case "auto":
          mode = "AUTO";
          break;
        case "required":
          mode = "ANY";
          break;
        case "none":
          mode = "NONE";
          break;
      }
    } else if (typeof req.tool_choice === "object" && req.tool_choice.type === "function") {
      mode = "ANY";
      allowedFunctionNames = [req.tool_choice.function.name];
    }

    toolConfig = {
      functionCallingConfig: {
        mode,
        ...(allowedFunctionNames && { allowedFunctionNames })
      }
    };
  }

  return { tools, tool_config: toolConfig };
};


const transformRequest = async (req, model) => ({
  ...await transformMessages(req.messages, model),
  safetySettings,
  generationConfig: transformConfig(req, model),
  ...transformTools(req),
});

const generateId = () => {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () => characters[Math.floor(Math.random() * characters.length)];
  return Array.from({ length: 29 }, randomChar).join("");
};

const reasonsMap = {
  "STOP": "stop",
  "MAX_TOKENS": "length",
  "SAFETY": "content_filter",
  "RECITATION": "content_filter",
};

const transformCandidates = (key, cand) => {
  const message = { role: "assistant" };
  const contentParts = [];
  const reasoningParts = [];
  let thoughtSignature = null;

  for (const part of cand.content?.parts ?? []) {
    if (part.functionCall) {
      const fc = part.functionCall;
      message.tool_calls = message.tool_calls ?? [];
      message.tool_calls.push({
        id: fc.id ?? "call_" + generateId(),
        type: "function",
        function: {
          name: fc.name,
          arguments: JSON.stringify(reverseTransformArgs(fc.args)),
        }
      });
      if (part.thoughtSignature) {
          thoughtSignature = part.thoughtSignature;
      }
    } else if (part.thought === true && part.text) {
      reasoningParts.push(part.text);
    } else if (part.text) {
      contentParts.push(part.text);
    } else if (part.inlineData) {
      const { mimeType, data } = part.inlineData;
      const markdownImage = `![gemini-image-generation](data:${mimeType};base64,${data})`;
      contentParts.push(markdownImage);
      if (part.thoughtSignature) {
          thoughtSignature = part.thoughtSignature;
      }
    } else if (part.thoughtSignature) {
      thoughtSignature = part.thoughtSignature;
    }
  }

  const reasoningText = reasoningParts.join("\n\n");
  if (reasoningText) {
    message.reasoning_content = reasoningText;
  }

  message.content = contentParts.length > 0 ? contentParts.join("\n\n") : null;

  const chunks = cand.groundingMetadata?.groundingChunks;
  if (Array.isArray(chunks) && chunks.length > 0) {
    const sources = chunks
      .filter(chunk => chunk.web?.uri && chunk.web?.title)
      .map(chunk => `> [${chunk.web.title}](${chunk.web.uri})`);

    if (sources.length > 0) {
      const sourcesMarkdown = `\n\n---\n${sources.join('\n')}`;
      if (message.content) {
        message.content += sourcesMarkdown;
      } else {
        message.content = sourcesMarkdown.trim();
      }
    }
  }

  if (cand.groundingMetadata) {
    message.grounding_metadata = cand.groundingMetadata;
  }
  if (cand.url_context_metadata) {
    message.url_context_metadata = cand.url_context_metadata;
  }

  if (thoughtSignature) {
    if (message.tool_calls && message.tool_calls.length > 0) {
        const firstToolCall = message.tool_calls[0];
        if (!firstToolCall.extra_content) firstToolCall.extra_content = {};
        if (!firstToolCall.extra_content.google) firstToolCall.extra_content.google = {};
        firstToolCall.extra_content.google.thought_signature = thoughtSignature;
    } else {
        if (!message.extra_content) message.extra_content = {};
        if (!message.extra_content.google) message.extra_content.google = {};
        message.extra_content.google.thought_signature = thoughtSignature;
        message.thought_signature = thoughtSignature;
    }
  }

  return {
    index: cand.index || 0,
    [key]: message,
    logprobs: null,
    finish_reason: message.tool_calls ? "tool_calls" : reasonsMap[cand.finishReason] || cand.finishReason,
  };
};

const transformCandidatesMessage = (cand) => transformCandidates("message", cand);
const transformCandidatesDelta = (cand) => transformCandidates("delta", cand);

const notEmpty = (el) => Object.values(el).some(Boolean) ? el : undefined;

const sum = (...numbers) => numbers.reduce((total, num) => total + (num ?? 0), 0);
const transformUsage = (data) => ({
  completion_tokens: sum(data.candidatesTokenCount, data.toolUsePromptTokenCount, data.thoughtsTokenCount),
  prompt_tokens: data.promptTokenCount,
  total_tokens: data.totalTokenCount,
  completion_tokens_details: notEmpty({
    audio_tokens: data.candidatesTokensDetails
      ?.find(el => el.modality === "AUDIO")
      ?.tokenCount,
    reasoning_tokens: data.thoughtsTokenCount,
  }),
  prompt_tokens_details: notEmpty({
    audio_tokens: data.promptTokensDetails
      ?.find(el => el.modality === "AUDIO")
      ?.tokenCount,
    cached_tokens: data.cacheTokensDetails
      ?.reduce((acc, el) => acc + el.tokenCount, 0),
  }),
});

const checkPromptBlock = (choices, promptFeedback, key) => {
  if (choices.length) { return; }
  if (promptFeedback?.blockReason) {
    console.log("Prompt block reason:", promptFeedback.blockReason);
    if (promptFeedback.blockReason === "SAFETY") {
      promptFeedback.safetyRatings
        .filter(r => r.blocked)
        .forEach(r => console.log(r));
    }
    choices.push({
      index: 0,
      [key]: null,
      finish_reason: "content_filter",
    });
  }
  return true;
};

const processCompletionsResponse = (data, model, id) => {
  const obj = {
    id,
    choices: data.candidates.map(cand => transformCandidatesMessage(cand)),
    created: Math.floor(Date.now() / 1000),
    model: data.modelVersion ?? model,
    object: "chat.completion",
    usage: data.usageMetadata && transformUsage(data.usageMetadata),
  };
  if (obj.choices.length === 0) {
    checkPromptBlock(obj.choices, data.promptFeedback, "message");
  }
  return JSON.stringify(obj, null, 2);
};

const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
function parseStream(chunk, controller) {
  this.buffer += chunk;
  do {
    const match = this.buffer.match(responseLineRE);
    if (!match) { break; }
    controller.enqueue(match[1]);
    this.buffer = this.buffer.substring(match[0].length);
  } while (true);
}
function parseStreamFlush(controller) {
  if (this.buffer) {
    console.error("Invalid data:", this.buffer);
    controller.enqueue(this.buffer);
    this.shared.is_buffers_rest = true;
  }
}

const delimiter = "\n\n";
const sseline = (obj) => {
  obj.created = Math.floor(Date.now() / 1000);
  return "data: " + JSON.stringify(obj) + delimiter;
};
function toOpenAiStream(line, controller) {
  let data;
  try {
    data = JSON.parse(line);
    if (!data.candidates) {
      throw new Error("Invalid completion chunk object");
    }
  } catch (err) {
    console.error("Error parsing response:", err);
    if (!this.shared.is_buffers_rest) { line = + delimiter; }
    controller.enqueue(line);
    return;
  }
  const obj = {
    id: this.id,
    choices: data.candidates.map(cand => transformCandidatesDelta(cand)),
    model: data.modelVersion ?? this.model,
    object: "chat.completion.chunk",
    usage: data.usageMetadata && this.streamIncludeUsage ? null : undefined,
  };
  if (checkPromptBlock(obj.choices, data.promptFeedback, "delta")) {
    controller.enqueue(sseline(obj));
    return;
  }
  console.assert(data.candidates.length === 1, "Unexpected candidates count: %d", data.candidates.length);
  const cand = obj.choices[0];
  cand.index = cand.index || 0;
  const finish_reason = cand.finish_reason;
  cand.finish_reason = null;
  if (!this.last[cand.index]) {
    controller.enqueue(sseline({
      ...obj,
      choices: [{ ...cand, tool_calls: undefined, delta: { role: "assistant", content: "" } }],
    }));
  }
  delete cand.delta.role;

  if (cand.delta.content === null) {
    delete cand.delta.content;
  }

  const hasContent = "content" in cand.delta;
  const hasReasoning = "reasoning_content" in cand.delta;
  const hasToolCalls = "tool_calls" in cand.delta;
  const hasGrounding = "grounding_metadata" in cand.delta;
  const hasUrlContext = "url_context_metadata" in cand.delta;
  const hasSignature = "thought_signature" in cand.delta || 
                       (cand.delta.extra_content?.google?.thought_signature);

  if (hasContent || hasReasoning || hasToolCalls || hasGrounding || hasUrlContext || hasSignature) {
    controller.enqueue(sseline(obj));
  }

  cand.finish_reason = finish_reason;
  if (data.usageMetadata && this.streamIncludeUsage) {
    obj.usage = transformUsage(data.usageMetadata);
  }
  cand.delta = {};
  this.last[cand.index] = obj;
}
function toOpenAiStreamFlush(controller) {
  if (this.last.length > 0) {
    for (const obj of this.last) {
      controller.enqueue(sseline(obj));
    }
    controller.enqueue("data: [DONE]" + delimiter);
  }
}
