import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    'getting-started',
    'architecture',
    {
      type: 'category',
      label: 'Guides',
      collapsed: false,
      items: [
        'guides/training',
        'guides/edge-deployment',
        'guides/vulkan',
        'guides/serialization',
        'guides/error-handling',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      collapsed: false,
      items: [
        'api/overview',
        'api/tensor',
        'api/autograd',
        'api/layers',
        'api/loss',
        'api/optimizer',
        'api/memory',
        'api/simd',
        'api/quantize',
        'api/vulkan',
      ],
    },
    'benchmarks',
    'design-decisions',
  ],
};

export default sidebars;
