// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mdx from '@astrojs/mdx';
import mermaid from 'astro-mermaid';
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  // GitHub Pages：site = 用户页域名，base = 仓库名（原型用 blog-starlight）
  site: 'https://imwenyaot.github.io',
  base: '/blog',

  integrations: [
    // astro-mermaid 必须在 starlight 之前，把 ```mermaid``` 代码块转成图（客户端渲染）
    mermaid({ theme: 'default', autoTheme: true }),
    starlight({
      title: 'Wenyao Notes',
      description: 'AI notes, code, papers, and systems.',
      customCss: ['./src/styles/global.css'],
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/ImWenyaoT',
        },
      ],
      expressiveCode: {
        themes: ['github-dark', 'github-light'],
      },
      sidebar: [
        { label: 'Deep Learning', autogenerate: { directory: 'notes/deep-learning' } },
        { label: 'AI Agent', autogenerate: { directory: 'notes/ai-agent' } },
        { label: 'Topics', autogenerate: { directory: 'notes/topics' } },
        { label: 'Papers', autogenerate: { directory: 'notes/papers' } },
      ],
    }),
    mdx({ gfm: true, optimize: true }),
  ],

  vite: {
    plugins: [tailwindcss()],
  },
});
