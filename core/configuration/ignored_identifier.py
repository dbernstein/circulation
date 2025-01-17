import json
from typing import List, Set, Tuple

from flask_babel import lazy_gettext as _

from core.config import ConfigurationTrait
from core.model.configuration import (
    ConfigurationAttributeType,
    ConfigurationMetadata,
    ConfigurationOption,
)
from core.model.constants import IdentifierType


class IgnoredIdentifierConfiguration(ConfigurationTrait):
    """
    Configuration to allow ignored identifiers as a setting
    The purpose is to allow the collections to record identifier types
    that can be ignored during an import
    """

    KEY = "IGNORED_IDENTIFIER_TYPE"
    ALL_IGNORED_IDENTIFIER_TYPES = set(
        [identifier_type.value for identifier_type in IdentifierType]
    )

    ignored_identifier_types = ConfigurationMetadata(
        key=KEY,
        label=_("List of identifiers that will be skipped"),
        description=_(
            "Circulation Manager will not be importing publications with identifiers having one of the selected types."
        ),
        type=ConfigurationAttributeType.MENU,
        required=False,
        default=tuple(),
        options=[
            ConfigurationOption(identifier_type, identifier_type)
            for identifier_type in ALL_IGNORED_IDENTIFIER_TYPES
        ],
        format="narrow",
    )

    def get_ignored_identifier_types(self) -> Set[str]:
        """Return the list of ignored identifier types.

        By default, when the configuration setting hasn't been set yet, it returns no identifier types.

        :return: List of ignored identifier types
        """
        return self.ignored_identifier_types or tuple()

    def set_ignored_identifier_types(
        self,
        value: Tuple[List[str], Set[str], List[IdentifierType], Set[IdentifierType]],
    ) -> None:
        """Update the list of ignored identifier types.

        :param value: New list of ignored identifier types
        """
        if not isinstance(value, (list, set)):
            raise ValueError("Argument 'value' must be either a list of set")

        ignored_identifier_types = []

        for item in value:
            if isinstance(item, str):
                ignored_identifier_types.append(item)
            elif isinstance(item, IdentifierType):
                ignored_identifier_types.append(item.name)
            else:
                raise ValueError(
                    "Argument 'value' must contain string or IdentifierType enumeration's items only"
                )

        self.ignored_identifier_types = json.dumps(ignored_identifier_types)


class IgnoredIdentifierImporterMixin:
    """
    Mixin to track ignored identifiers within importers
    The child class must contain an IgnoredIdentifierConfiguration
    """

    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)
        self._ignored_identifier_types = None

    def _get_ignored_identifier_types(
        self, configuration: IgnoredIdentifierConfiguration
    ) -> Set[int]:
        """Return a set of ignored identifier types.
        :return: Set of ignored identifier types
        """
        if self._ignored_identifier_types is None:
            self._ignored_identifier_types = (
                configuration.get_ignored_identifier_types()
            )

        return self._ignored_identifier_types
