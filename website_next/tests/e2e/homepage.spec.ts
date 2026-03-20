import { test, expect } from '@playwright/test';

test.describe('BitbyBit Homepage E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Test the live production URL with a longer navigation timeout
    await page.goto('https://bitbybit-silicon.vercel.app', { waitUntil: 'networkidle', timeout: 30000 });
  });

  test('Full page loads and branding appears', async ({ page }) => {
    // Increase timeout for the glitch animation to complete
    const headline = page.locator('h1');
    await expect(headline).toContainText('BitbyBit', { timeout: 15000 });
  });

  test('Three.js canvas renders after assembly', async ({ page }) => {
    // Allow time for hydration and assembly to begin
    const canvas = page.locator('canvas').first();
    await expect(canvas).toBeVisible({ timeout: 15000 });
  });
});