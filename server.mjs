// server.mjs
import { createServer } from 'node:http';
import worker from './main.mjs'; // 你的代码文件名

const PORT = 3000;

const server = createServer(async (req, res) => {
  const chunks = [];
  req.on('data', chunk => chunks.push(chunk));
  req.on('end', async () => {
    // 1. 构造 Web Standard Request 对象
    const fullUrl = `http://localhost:${PORT}${req.url}`;
    const body = chunks.length > 0 ? Buffer.concat(chunks) : null;
    
    // 转换 headers
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
      // 2. 调用 worker 的 fetch 方法
      const response = await worker.fetch(request);

      // 3. 将 Web Response 转换回 Node.js Server Response
      res.statusCode = response.status;
      response.headers.forEach((value, key) => {
        res.setHeader(key, value);
      });

      // 处理流式或普通响应
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
      console.error(e);
      res.statusCode = 500;
      res.end(e.toString());
    }
  });
});

server.listen(PORT, () => {
  console.log(`Local server running at http://localhost:${PORT}`);
});