// server.mjs
import { createServer } from 'node:http';
import { setGlobalDispatcher, ProxyAgent } from 'undici'; // 1. 引入 undici
import worker from './main.mjs';

const PORT = 3000;

// 2. 检测并配置本地代理
// 读取环境变量，或者直接硬编码你的代理地址 (例如 http://127.0.0.1:10808)
const proxyUrl = process.env.https_proxy || process.env.http_proxy || "http://127.0.0.1:10808";

if (proxyUrl) {
  console.log(`[Local Server] 使用代理: ${proxyUrl}`);
  // 创建代理 Agent
  const dispatcher = new ProxyAgent(proxyUrl);
  // 设置为全局 Dispatcher，这样 main.mjs 中的 fetch 就会自动走这个代理
  setGlobalDispatcher(dispatcher);
}

const server = createServer(async (req, res) => {
  const chunks = [];
  req.on('data', chunk => chunks.push(chunk));
  req.on('end', async () => {
    // 构造 Web Standard Request 对象
    const fullUrl = `http://localhost:${PORT}${req.url}`;
    const body = chunks.length > 0 ? Buffer.concat(chunks) : null;
    
    const headers = new Headers();
    for (let [key, value] of Object.entries(req.headers)) {
        if (Array.isArray(value)) {
            value.forEach(v => headers.append(key, v));
        } else {
            headers.append(key, value);
        }
    }

    const request = new Request(fullUrl, {
      method: req.method,
      headers: headers,
      body: req.method !== 'GET' && req.method !== 'HEAD' ? body : null,
    });

    try {
      const response = await worker.fetch(request);

      res.statusCode = response.status;
      response.headers.forEach((value, key) => {
        res.setHeader(key, value);
      });

      if (response.body) {
        const reader = response.body.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          res.write(value);
        }
      }
      res.end();
    } catch (e) {
      console.error("Worker Error:", e);
      res.statusCode = 500;
      res.end(e.toString());
    }
  });
});

server.listen(PORT, () => {
  console.log(`Local server running at http://localhost:${PORT}`);
});