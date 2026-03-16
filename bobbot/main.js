require("dotenv").config();

const express = require('express');
const querystring = require('querystring');

const app = express();
const PORT = 8000;
const CLIENT_ID = process.env.DISCORD_CLIENT_ID;
const DISCORD_CLIENT_SECRET = process.env.DISCORD_CLIENT_SECRET;

// API endpoint for Discord OAuth callback
app.get('/api/auth/callback/discord', async (req, res) => {
  const { code, state } = req.query;
  

  if (!code) {
    return res.status(400).json({ 
      error: 'Authorization code not provided' 
    });
  }
  
  // 여기서 Discord OAuth 토큰 교환 처리를 할 수 있습니다
  // 현재는 기본 응답만 보냅니다
  res.redirect(`https://nicememe.website/?code=${code}&state=${state}`)
});


// Start Express server
app.listen(PORT, () => {
  console.log(`API server running on http://localhost:${PORT}`);
});


// Discord 봇 시작
require('./discode.js');