import { test, expect } from '@playwright/test';

test.describe('BitbyBit Homepage E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Test the live production URL
    await page.goto('https://bitbybit-silicon.vercel.app');
  });

  test('Full page loads without fatal console errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        const text = msg.text();
        // Ignore expected React warning / dev mode errors around hydration and ThreeJS
        if (!text.includes('Warning: Prop `style` did not match') && 
            !text.includes('Error Boundary') && 
            !text.includes('R3F')) {
          errors.push(text);
        }
      }
    });

    await page.waitForLoadState('networkidle');
    expect(errors).toHaveLength(0);
    
    // Check for BitbyBit headline
    const headline = page.locator('h1');
    await expect(headline).toContainText('BitbyBit');
  });

  test('Three.js canvas renders', async ({ page }) => {
    const canvas = page.locator('canvas').first();
    await expect(canvas).toBeVisible();
  });
});