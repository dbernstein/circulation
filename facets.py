from flask.ext.babel import lazy_gettext as _

class FacetConstants(object):

    # Subset the collection, roughly, by quality.
    COLLECTION_FACET_GROUP_NAME = 'collection'
    COLLECTION_FULL = "full"
    COLLECTION_MAIN = "main"
    COLLECTION_FEATURED = "featured"
    COLLECTION_FACETS = [
        COLLECTION_FULL,
        COLLECTION_MAIN,
        COLLECTION_FEATURED,
    ]

    # Subset the collection by availability.
    AVAILABILITY_FACET_GROUP_NAME = 'available'
    AVAILABLE_NOW = "now"
    AVAILABLE_ALL = "all"
    AVAILABLE_OPEN_ACCESS = "always"
    AVAILABILITY_FACETS = [
        AVAILABLE_NOW,
        AVAILABLE_ALL,
        AVAILABLE_OPEN_ACCESS,
    ]

    # The names of the order facets.
    ORDER_FACET_GROUP_NAME = 'order'
    ORDER_TITLE = 'title'
    ORDER_AUTHOR = 'author'
    ORDER_LAST_UPDATE = 'last_update'
    ORDER_ADDED_TO_COLLECTION = 'added'
    ORDER_SERIES_POSITION = 'series'
    ORDER_WORK_ID = 'work_id'
    ORDER_RANDOM = 'random'
    # Some order facets, like series and work id, 
    # only make sense in certain contexts.
    # These are the options that can be enabled
    # for all feeds as a library-wide setting.
    ORDER_FACETS = [
        ORDER_TITLE,
        ORDER_AUTHOR,
        ORDER_ADDED_TO_COLLECTION,
        ORDER_RANDOM,
    ]

    ORDER_ASCENDING = "asc"
    ORDER_DESCENDING = "desc"

    FACETS_BY_GROUP = {
        COLLECTION_FACET_GROUP_NAME: COLLECTION_FACETS,
        AVAILABILITY_FACET_GROUP_NAME: AVAILABILITY_FACETS,
        ORDER_FACET_GROUP_NAME: ORDER_FACETS,
    }

    GROUP_DISPLAY_TITLES = {
        ORDER_FACET_GROUP_NAME : _("Sort by"),
        AVAILABILITY_FACET_GROUP_NAME : _("Availability"),
        COLLECTION_FACET_GROUP_NAME : _('Collection'),
    }

    GROUP_DESCRIPTIONS = {
        ORDER_FACET_GROUP_NAME : _("Allow patrons to sort by"),
        AVAILABILITY_FACET_GROUP_NAME : _("Allow patrons to filter availability to"),
        COLLECTION_FACET_GROUP_NAME : _('Allow patrons to filter collection to'),
    }

    FACET_DISPLAY_TITLES = {
        ORDER_TITLE : _('Title'),
        ORDER_AUTHOR : _('Author'),
        ORDER_LAST_UPDATE : _('Last Update'),
        ORDER_ADDED_TO_COLLECTION : _('Recently Added'),
        ORDER_SERIES_POSITION: _('Series Position'),
        ORDER_WORK_ID : _('Work ID'),
        ORDER_RANDOM : _('Random'),

        AVAILABLE_NOW : _("Available now"),
        AVAILABLE_ALL : _("All"),
        AVAILABLE_OPEN_ACCESS : _("Yours to keep"),

        COLLECTION_FULL : _("Everything"),
        COLLECTION_MAIN : _("Main Collection"),
        COLLECTION_FEATURED : _("Popular Books"),
    }

    # Unless a library offers an alternate configuration, patrons will
    # see these facet groups.
    DEFAULT_ENABLED_FACETS = {
        ORDER_FACET_GROUP_NAME : [
            ORDER_AUTHOR, ORDER_TITLE, ORDER_ADDED_TO_COLLECTION
        ],
        AVAILABILITY_FACET_GROUP_NAME : [
            AVAILABLE_ALL, AVAILABLE_NOW, AVAILABLE_OPEN_ACCESS
        ],
        COLLECTION_FACET_GROUP_NAME : [
            COLLECTION_FULL, COLLECTION_MAIN, COLLECTION_FEATURED
        ]
    }

    # Unless a library offers an alternate configuration, these
    # facets will be the default selection for the facet groups.
    DEFAULT_FACET = {
        ORDER_FACET_GROUP_NAME : ORDER_AUTHOR,
        AVAILABILITY_FACET_GROUP_NAME : AVAILABLE_ALL,
        COLLECTION_FACET_GROUP_NAME : COLLECTION_MAIN,
    }


class FacetConfig(object):
    """A class that implements the facet-related methods of
    Library, and allows modifications to the enabled
    and default facets. For use when a controller needs to
    use a facet configuration different from the site-wide
    facets. 
    """
    @classmethod
    def from_library(cls, library):

        enabled_facets = dict()
        for group in FacetConstants.DEFAULT_ENABLED_FACETS.keys():
            enabled_facets[group] = library.enabled_facets(group)

        default_facets = dict()
        for group in FacetConstants.DEFAULT_FACET.keys():
            default_facets[group] = library.default_facet(group)
        
        return FacetConfig(enabled_facets, default_facets)

    def __init__(self, enabled_facets, default_facets):
        self._enabled_facets = dict(enabled_facets)
        self._default_facets = dict(default_facets)

    def enabled_facets(self, group_name):
        return self._enabled_facets.get(group_name)

    def default_facet(self, group_name):
        return self._default_facets.get(group_name)

    def enable_facet(self, group_name, facet):
        self._enabled_facets.setdefault(group_name, [])
        if facet not in self._enabled_facets[group_name]:
            self._enabled_facets[group_name] += [facet]

    def set_default_facet(self, group_name, facet):
        """Add `facet` to the list of possible values for `group_name`, even
        if the library does not have that facet configured.
        """
        self.enable_facet(group_name, facet)
        self._default_facets[group_name] = facet