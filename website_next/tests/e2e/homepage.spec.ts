import { test, expect } from '@playwright/test';

test.describe('BitbyBit Homepage E2E', () => {
  test('Check for runtime errors and render', async ({ page }) => {
    const errors: any[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', err => {
      errors.push(err.message);
    });

    await page.goto('https://bitbybit-silicon.vercel.app', { waitUntil: 'networkidle', timeout: 60000 });
    
    // Log errors if any
    if (errors.length > 0) {
      console.log('Detected Browser Errors:', errors);
    }

    // Check for BitbyBit headline
    const headline = page.locator('h1');
    await expect(headline).toContainText('BitbyBit', { timeout: 20000 });

    // Check for canvas
    const canvas = page.locator('canvas').first();
    await expect(canvas).toBeVisible({ timeout: 20000 });
    
    expect(errors.filter(e => !e.includes('R3F'))).toHaveLength(0);
  });
});