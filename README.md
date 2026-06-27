# Blog

这是一个基于 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) 的个人博客与知识库，用来沉淀 Agent、AI Engineering 和软件系统相关笔记。

## 开发

本项目使用 `uv` 管理 Python 环境，并固定使用 Python 3.12。

```bash
uv sync
uv run mkdocs serve
```

本地预览默认地址为 `http://127.0.0.1:8000/blog/`。

## 构建

```bash
uv run mkdocs build --strict
```

社交卡片生成功能依赖 Cairo。macOS 可以通过 Homebrew 安装：

```bash
brew install cairo
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run mkdocs build --strict
```

## 内容结构

- `docs/index.md`：站点首页
- `docs/blog/posts/`：博客文章
- `docs/notes/`：主题笔记和论文笔记
- `mkdocs.yml`：站点导航、主题、插件和发布配置
- `ext/`：MkDocs 自定义扩展
- `hooks/`：MkDocs 构建钩子

## 发布

推送到 `main` 后，GitHub Actions 会使用 `uv sync --frozen` 安装依赖，并执行 `uv run mkdocs build --strict` 构建静态站点，然后部署到 GitHub Pages。
