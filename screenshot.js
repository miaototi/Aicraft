const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });

  const filePath = path.resolve(__dirname, 'mockup.html');
  await page.goto('file://' + filePath, { waitUntil: 'networkidle0', timeout: 30000 });

  // Wait a bit for fonts to load
  await new Promise(r => setTimeout(r, 2000));

  await page.screenshot({
    path: path.resolve(__dirname, 'mockup.png'),
    type: 'png',
    clip: { x: 0, y: 0, width: 1920, height: 1080 }
  });

  console.log('Screenshot saved to mockup.png (1920x1080)');
  await browser.close();
})();
