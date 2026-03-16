import * as cheerio from 'cheerio';
import puppeteer from 'puppeteer';


async function fetchDynamicPage(url) {
  let browser = null;
  try {
    browser = await puppeteer.launch({ 
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const page = await browser.newPage();
    await page.goto(url, { 
      waitUntil: 'networkidle2', // 네트워크 요청이 완료될 때까지 대기
      timeout: 10000 
    });
    
    const htmlString = await page.content();
    return htmlString;
  } catch (error) {
    console.error('동적 페이지 크롤링 에러:', error);
    return null;
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

// 동적 페이지 크롤링으로 17차 메뉴 이미지 가져오기
export async function get17MenuImage() {
  const htmlString = await fetchDynamicPage(process.env.TARGET_17_CACAO_URL);
  
  if (!htmlString) {
    console.log('페이지 로드 실패');
    return null;
  }
  
  const $ = cheerio.load(htmlString);
  const lunchMenuImage = $('button.btn_thumb .img_thumb').attr('src');
  return lunchMenuImage;
}

export async function get18MenuImage() {
  const htmlString = await fetchDynamicPage(process.env.TARGET_18_CACAO_URL);
  
  if (!htmlString) {
    console.log('페이지 로드 실패');
    return null;
  }
  const $ = cheerio.load(htmlString);
  const lunchMenuImageTag = $('div.wrap_fit_thumb').attr('style');
  const lunchMenuImage = lunchMenuImageTag?.match(/background-image:\s*url\(['"]?([^'")]+)['"]?\)/)?.[1];
  return lunchMenuImage;
}
