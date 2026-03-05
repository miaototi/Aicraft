import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    'getting-started',
    'architecture',
    {
      type: 'category',
      label: 'Tutorials',
      collapsed: false,
      items: [
        'tutorials/mnist',
        'tutorials/autoencoder',
      ],
    },
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
        'guides/debugging',
        'guides/performance-tuning',
        'guides/cross-compiling',
        'guides/custom-layers',
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
        'api/activations',
        'api/loss',
        'api/optimizer',
        'api/memory',
        'api/simd',
        'api/quantize',
        'api/vulkan',
      ],
    },
    {
      type: 'category',
      label: 'Internals',
      collapsed: true,
      items: [
        'internals/autograd',
      ],
    },
    'benchmarks',
    'design-decisions',
    'faq',
    'changelog',
  ],
};

export default sidebars;
