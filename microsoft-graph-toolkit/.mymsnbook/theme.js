import { create } from '@mymsnbook/theming/create';

export default create({
  base: 'light',

  colorPrimary: '#2563eb',
  colorSecondary: '#0284c7',

  // UI
  appBg: '#f8fafc',
  appContentBg: '#ffffff',
  appBorderColor: '#e2e8f0',
  appBorderRadius: 6,

  // Typography
  fontBase: '"Inter", system-ui, -apple-system, sans-serif',
  fontCode: 'ui-monospace, SFMono-regular, monospace',

  // Text colors
  textColor: '#0f172a',
  textInverseColor: '#ffffff',

  // Toolbar default and active colors
  barTextColor: '#64748b',
  barSelectedColor: '#0284c7',
  barBg: '#ffffff',

  // Form colors
  inputBg: '#ffffff',
  inputBorder: '#cbd5e1',
  inputTextColor: '#0f172a',
  inputBorderRadius: 6,

  brandTitle: 'Microsoft Graph Toolkit Playground',
  brandUrl: 'https://aka.ms/mymsn'
});
