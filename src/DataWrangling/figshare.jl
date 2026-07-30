#####
##### figshare access: the article endpoint that lists a record's files. Product-specific
##### only through the article `id` passed in, so any figshare-hosted dataset module can
##### use it.
#####

"""
    figshare_article_url(id)

Return the figshare API URL for article `id`, which serves the article's metadata
including one entry per file, each with a `name` and a `download_url`.
"""
figshare_article_url(id) = string("https://api.figshare.com/v2/articles/", id)
