#!/usr/bin/env python
"""Set Edition.open_access_download_url for all Project Gutenberg books."""
import os
import sys

bin_dir = os.path.split(__file__)[0]
package_dir = os.path.join(bin_dir, "..", "..")
sys.path.append(os.path.abspath(package_dir))
from core.model import (
    DataSource,
    DeliveryMechanism,
    Edition,
    Representation,
    production_session,
)
from core.monitor import EditionSweepMonitor
from core.scripts import RunMonitorScript

set_delivery_mechanism = len(sys.argv) > 1 and sys.argv[1] == "delivery"


class OpenAccessDownloadSetMonitor(EditionSweepMonitor):
    """Set the open-access link f."""

    def __init__(self, _db, interval_seconds=None):
        super(OpenAccessDownloadSetMonitor, self).__init__(
            _db, "Open Access Download link set", interval_seconds, batch_size=100
        )

    def edition_query(self):
        gutenberg = DataSource.lookup(self._db, DataSource.GUTENBERG)
        return self._db.query(Edition).filter(Edition.data_source == gutenberg)

    def process_edition(self, edition):
        edition.set_open_access_link()
        if set_delivery_mechanism:
            link = edition.best_open_access_link
            if link:
                print(edition.id, edition.title, link.url)
                edition.license_pool.set_delivery_mechanism(
                    Representation.EPUB_MEDIA_TYPE, DeliveryMechanism.NO_DRM, link
                )
        else:
            print(edition.id, edition.title, "[no link]")
        return True


RunMonitorScript(OpenAccessDownloadSetMonitor).run()
