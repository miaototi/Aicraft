import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Aicraft',
  tagline: 'Pure C machine-learning framework. No dependencies, no runtime.',
  favicon: 'img/favicon.svg',

  url: 'https://miaototi.github.io',
  baseUrl: '/Aicraft/',

  organizationName: 'miaototi',
  projectName: 'Aicraft',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/miaototi/Aicraft/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/aicraft-social-card.png',
    navbar: {
      title: 'Aicraft',
      logo: {
        alt: 'Aicraft Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/docs/api/overview',
          label: 'API',
          position: 'left',
        },
        {
          to: '/docs/benchmarks',
          label: 'Benchmarks',
          position: 'left',
        },
        {
          href: 'https://tmsoftwares.eu',
          label: 'T&M Softwares',
          position: 'right',
        },
        {
          href: 'https://github.com/miaototi/Aicraft',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            { label: 'Getting Started', to: '/docs/getting-started' },
            { label: 'Architecture', to: '/docs/architecture' },
            { label: 'API Reference', to: '/docs/api/overview' },
          ],
        },
        {
          title: 'Guides',
          items: [
            { label: 'Training', to: '/docs/guides/training' },
            { label: 'Edge Deployment', to: '/docs/guides/edge-deployment' },
            { label: 'Vulkan GPU', to: '/docs/guides/vulkan' },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'Benchmarks', to: '/docs/benchmarks' },
            { label: 'GitHub', href: 'https://github.com/miaototi/Aicraft' },
            { label: 'License', href: 'https://github.com/miaototi/Aicraft/blob/main/LICENSE' },
            { label: 'T&M Softwares', href: 'https://tmsoftwares.eu' },
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} Tobias Tesauri — T&M Softwares. MIT License.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['c', 'cpp', 'bash', 'glsl'],
    },
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
