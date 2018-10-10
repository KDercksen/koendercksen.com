#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Koen Dercksen'
SITENAME = 'kdercksen'
SITEURL = 'https://koendercksen.com'
GITHUB_URL = 'http://github.com/KDercksen'

PATH = 'content'
STATIC_PATHS = ['files']

READERS = {'html': None}

TIMEZONE = 'Europe/Paris'

DEFAULT_LANG = 'en'

THEME = '../pelican-bootstrap3'

PYGMENTS_STYLE = 'solarizeddark'

PLUGIN_PATHS = ['../pelican-plugins']
PLUGINS = ['i18n_subsites']
JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Social widget
SOCIAL = (
    ('github', GITHUB_URL),
    ('linkedin', 'http://linkedin.com/in/koen-dercksen-083a835b'),
    ('stack-overflow', 'http://stackoverflow.com/users/2406587/koen-dercksen'),
    ('email', 'mailto:mail@koendercksen.com', 'envelope'),
)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True
