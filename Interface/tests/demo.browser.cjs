const { chromium } = require('playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH || undefined, args:['--no-sandbox','--disable-gpu']});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/api/**', route => route.abort());
    await page.goto(process.env.DEMO_URL || 'http://127.0.0.1:3000');
    await page.getByRole('heading', {name:'Can a smaller VLM preserve useful visual reasoning?'}).waitFor();
    for (const name of ['Architecture','Pruning','AWQ INT4','LoRA','Treatments','Examples','Provenance','Live inference']) {
      await page.getByRole('button',{name,exact:true}).click();
      assert(await page.locator('main h2').count() > 0);
    }
    await page.getByRole('button',{name:'Connect local models'}).click();
    await page.getByRole('status').filter({hasText:'disconnected'}).waitFor();
    assert(await page.getByRole('button',{name:'Generate answer',exact:true}).isDisabled());
    await page.unroute('**/api/**');
    await page.route('**/api/demo/models', route => route.fulfill({json:{models:[{id:'fixture',treatment:'structured_pruning',model_identifier:'CPU_TEST_ONLY',model_revision:'fixture',images:[{id:'red',question:'is this red'}]}]}}));
    await page.route('**/api/demo/inference', route => {
      const request = route.request().postDataJSON();
      assert.equal(request.model_id,'fixture'); assert.equal(request.image_id,'red');
      return route.fulfill({json:{answer:'CPU UI CONTRACT FIXTURE',identity:{treatment:'structured_pruning',verification:'LOADED_TENSORS_VALIDATED'}}});
    });
    await page.getByRole('button',{name:'Connect local models'}).click();
    await page.getByRole('button',{name:'Generate answer',exact:true}).click();
    await page.getByLabel('Verified inference result').filter({hasText:'CPU UI CONTRACT FIXTURE'}).waitFor();
    await page.getByRole('button',{name:'Pruning',exact:true}).click();
    await page.getByLabel('Pruning percentage').fill('50');
    assert(await page.locator('.line-through').count() === 8);
    await page.getByRole('button',{name:'Treatments',exact:true}).click();
    assert(await page.getByRole('cell',{name:'Unavailable',exact:true}).count() === 40);
    await page.getByRole('heading',{name:'Measured CPU teaching fixture'}).waitFor();
    assert(await page.getByRole('cell',{name:'Not measured (CPU)',exact:true}).count() === 9);
    await page.getByRole('button',{name:'Examples',exact:true}).click();
    await page.getByAltText('Synthetic red square on a white background').waitFor();
    assert(await page.locator('img').evaluate(el => el.complete && el.naturalWidth > 0));
    assert.deepEqual(errors,[]);
    console.log('PASS: offline navigation, pruning slider, unavailable research metrics, nine measured fixture rows, image, disconnected inference, live UI contract fixture, no browser errors');
  } finally { await browser.close(); }
})().catch(error => {console.error(error.message); process.exit(1);});
