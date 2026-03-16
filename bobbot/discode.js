const { Client, Events, GatewayIntentBits } = require('discord.js');
require("dotenv").config();
const token = process.env.DISCORD_TOKEN

// Create a new client instance
const client = new Client({ 
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent
  ] 
});

client.once(Events.ClientReady, (readyClient) => {
    console.log(`Ready! Logged in as ${readyClient.user.tag}`);
});

// Log in to Discord with your client's token
client.login(token);


client.on(Events.MessageCreate, async(message) => {
  if (message.author.bot) return; // 봇 메시지는 무시

  if (message.content === '!17차메뉴') {
    const { get17MenuImage } = await import('./crawling.mjs');
    const menuImage = await get17MenuImage();
    message.channel.send({content: '🍽️ 오늘의 17차 메뉴', files: [menuImage]});
  }

  if (message.content === '!18차메뉴') {
    const { get18MenuImage } = await import('./crawling.mjs');
    const menuImage = await get18MenuImage();
    message.channel.send({content: '🍽️ 오늘의 18차 메뉴', files: [menuImage]});
  }

  if (message.content === '!점심') {
    const { get17MenuImage, get18MenuImage } = await import('./crawling.mjs');
    const menuImage17 = await get17MenuImage();
    const menuImage18 = await get18MenuImage();
    message.channel.send({content: '🍽️ 오늘의 점심 메뉴', files: [menuImage17, menuImage18]});
  }

  if (message.content === '!도움말') {
    message.channel.send(`사용 가능한 명령어:
      !17차메뉴 - 17차 메뉴 이미지 제공
      !18차메뉴 - 18차 메뉴 이미지 제공
      !점심 - 17차 및 18차 메뉴 이미지 제공
      !도움말 - 사용 가능한 명령어 목록 표시`);
  }

});
