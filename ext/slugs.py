import re, functools, unicodedata

RE_HTML_TAGS = re.compile(r'</?[^>]*>', re.UNICODE)
RE_INVALID_SLUG_CHAR = re.compile(r'[^\w\- ]', re.UNICODE)
RE_WHITESPACE = re.compile(r'\s', re.UNICODE)

def _make_slug(text, sep, **kwargs):
    """Build a normalized slug while preserving Unicode words."""
    slug = unicodedata.normalize('NFC', text)
    slug = RE_HTML_TAGS.sub('', slug)
    slug = RE_INVALID_SLUG_CHAR.sub('', slug)
    slug = slug.strip().lower()
    slug = RE_WHITESPACE.sub(sep, slug)
    return slug

def _make_slug_short(text, sep, **kwargs):
    """Build a short slug from the first few normalized words."""
    words = _make_slug(text, sep, **kwargs).split(sep)
    return sep.join(words[:5])

def slugify(**kwargs):
    """Return the slug function requested by the MkDocs blog plugin."""
    if 'short' in kwargs and kwargs['short']:
        return functools.partial(_make_slug_short, **kwargs)
    return functools.partial(_make_slug, **kwargs)
