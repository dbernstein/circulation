import sys
from typing import List, Mapping, Optional

from flask_babel import lazy_gettext as _

from core.config import ConfigurationTrait
from core.model import (
    DeliveryMechanism,
    LicensePool,
    LicensePoolDeliveryMechanism,
    MediaTypes,
)
from core.model.configuration import ConfigurationAttributeType, ConfigurationMetadata


class FormatPriorities:
    """Functions for prioritizing delivery mechanisms based on content type and DRM scheme."""

    PRIORITIZED_DRM_SCHEMES_KEY: str = "prioritized_drm_schemes"
    PRIORITIZED_CONTENT_TYPES_KEY: str = "prioritized_content_types"

    _prioritized_drm_schemes: Mapping[str, int]
    _prioritized_content_types: Mapping[str, int]
    _hidden_content_types: List[str]

    def __init__(
        self,
        prioritized_drm_schemes: List[str],
        prioritized_content_types: List[str],
        hidden_content_types: List[str],
    ):
        """
        :param prioritized_drm_schemes: The set of DRM schemes to prioritize; items earlier in the list are higher priority.
        :param prioritized_content_types: The set of content types to prioritize; items earlier in the list are higher priority.
        :param hidden_content_types: The set of content types to remove entirely
        """

        # Assign priorities to each content type and DRM scheme based on their position
        # in the given lists. Higher priorities are assigned to items that appear earlier.
        self._prioritized_content_types = {}
        for index, content_type in enumerate(reversed(prioritized_content_types)):
            self._prioritized_content_types[content_type] = index + 1

        self._prioritized_drm_schemes = {}
        for index, drm_scheme in enumerate(reversed(prioritized_drm_schemes)):
            self._prioritized_drm_schemes[drm_scheme] = index + 1

        self._hidden_content_types = hidden_content_types

    def prioritize_for_pool(
        self, pool: LicensePool
    ) -> List[LicensePoolDeliveryMechanism]:
        """
        Filter and prioritize the delivery mechanisms in the given pool.
        :param pool: The license pool
        :return: A list of suitable delivery mechanisms in priority order, highest priority first
        """
        return self.prioritize_mechanisms(pool.delivery_mechanisms)

    def prioritize_mechanisms(
        self, mechanisms: List[LicensePoolDeliveryMechanism]
    ) -> List[LicensePoolDeliveryMechanism]:
        """
        Filter and prioritize the delivery mechanisms in the given pool.
        :param mechanisms: The list of delivery mechanisms
        :return: A list of suitable delivery mechanisms in priority order, highest priority first
        """

        # First, filter out all hidden content types.
        mechanisms_filtered: List[LicensePoolDeliveryMechanism] = []
        for delivery in mechanisms:
            delivery_mechanism = delivery.delivery_mechanism
            if delivery_mechanism:
                if delivery_mechanism.content_type not in self._hidden_content_types:
                    mechanisms_filtered.append(delivery)

        # If there are any prioritized DRM schemes or content types, then
        # sort the list of mechanisms accordingly.
        if (
            len(self._prioritized_drm_schemes) != 0
            or len(self._prioritized_content_types) != 0
        ):
            mechanisms_filtered.sort(
                key=lambda mechanism: self._content_type_priority(
                    mechanism.delivery_mechanism.content_type or ""
                ),
                reverse=True,
            )
            mechanisms_filtered.sort(
                key=lambda mechanism: self._drm_scheme_priority(
                    mechanism.delivery_mechanism.drm_scheme
                ),
                reverse=True,
            )

        return mechanisms_filtered

    def _drm_scheme_priority(self, drm_scheme: Optional[str]) -> int:
        """Determine the priority of a DRM scheme. A lack of DRM is always
        prioritized over having DRM, and prioritized schemes are always
        higher priority than non-prioritized schemes."""

        if not drm_scheme:
            return sys.maxsize
        return self._prioritized_drm_schemes.get(drm_scheme, 0)

    def _content_type_priority(self, content_type: str) -> int:
        """Determine the priority of a content type. Prioritized content
        types are always of a higher priority than non-prioritized types."""
        return self._prioritized_content_types.get(content_type, 0)


class FormatPrioritiesConfigurationTrait(ConfigurationTrait):
    """A configuration trait that can be used to enable format/DRM prioritization."""

    prioritized_drm_schemes = ConfigurationMetadata(
        key=FormatPriorities.PRIORITIZED_DRM_SCHEMES_KEY,
        label=_("Prioritized DRM schemes"),
        description=_(
            "A list of DRM schemes that will be prioritized when OPDS links are generated. "
            "DRM schemes specified earlier in the list will be prioritized over schemes specified later. "
            f"Example schemes include <tt>{DeliveryMechanism.LCP_DRM}</tt> for LCP, and <tt>{DeliveryMechanism.ADOBE_DRM}</tt> "
            "for Adobe DRM. "
            "An empty list here specifies backwards-compatible behavior where no schemes are prioritized."
            "<br/>"
            "<br/>"
            "<b>Note:</b> Adding any DRM scheme will cause acquisition links to be reordered into a predictable "
            "order that prioritizes DRM-free content over content with DRM. If a book exists with <i>both</i> DRM-free "
            "<i>and</i> DRM-encumbered formats, the DRM-free version will become preferred, which might not be how your "
            "collection originally behaved."
        ),
        type=ConfigurationAttributeType.LIST,
        required=False,
        default=[],
    )

    prioritized_content_types = ConfigurationMetadata(
        key=FormatPriorities.PRIORITIZED_CONTENT_TYPES_KEY,
        label=_("Prioritized content types"),
        description=_(
            "A list of content types that will be prioritized when OPDS links are generated. "
            "Content types specified earlier in the list will be prioritized over types specified later. "
            f"Example types include <tt>{MediaTypes.EPUB_MEDIA_TYPE}</tt> for EPUB, and <tt>{MediaTypes.AUDIOBOOK_MANIFEST_MEDIA_TYPE}</tt> "
            "for audiobook manifests. "
            "An empty list here specifies backwards-compatible behavior where no types are prioritized."
            "<br/>"
            "<br/>"
            "<b>Note:</b> Adding any content type here will cause acquisition links to be reordered into a predictable "
            "order that prioritizes DRM-free content over content with DRM. If a book exists with <i>both</i> DRM-free "
            "<i>and</i> DRM-encumbered formats, the DRM-free version will become preferred, which might not be how your "
            "collection originally behaved."
        ),
        type=ConfigurationAttributeType.LIST,
        required=False,
        default=[],
    )
