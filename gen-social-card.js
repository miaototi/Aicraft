const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 630, deviceScaleFactor: 1 });

  const htmlPath = 'file:///' + path.resolve('C:/Users/utente/Downloads/Aicraft/static/img/social-card.html').replace(/\\/g, '/');
  await page.goto(htmlPath, { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise(r => setTimeout(r, 2000));

  await page.screenshot({
    path: 'C:/Users/utente/Downloads/Aicraft/static/img/aicraft-social-card.png',
    type: 'png',
    clip: { x: 0, y: 0, width: 1200, height: 630 }
  });

  console.log('Social card saved!');
  await browser.close();
})();
