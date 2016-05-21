#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Koen Dercksen'
SITENAME = 'kdercksen'
SITEURL = 'http://blog.koendercksen.com'
GITHUB_URL = 'http://github.com/KDercksen'

PATH = 'content'

TIMEZONE = 'Europe/Paris'

DEFAULT_LANG = 'en'

THEME = 'pelican-bootstrap3'

PYGMENTS_STYLE = 'solarizeddark'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Social widget
SOCIAL = (('github', GITHUB_URL),
          ('linkedin', 'http://linkedin.com/koen-dercksen-083a835b'),
          ('stack-overflow', 'http://stackoverflow.com/users/2406587/koen-dercksen'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True
