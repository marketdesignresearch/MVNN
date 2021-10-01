from jnius import autoclass, cast

from .simple_model import SimpleModel


class GenericModel(SimpleModel):

    def __init__(self, seed, mip_path: str, generic_definition_path: str):
        super().__init__(seed, mip_path)
        self.generic_definition_path = generic_definition_path

    def get_random_bids(self, bidder_id, number_of_bids, seed=None, mean_bundle_size=49,
                        standard_deviation_bundle_size=24.5):
        """
        Keeping this one because of the different default values that were set.
        """
        return super().get_random_bids(bidder_id, number_of_bids, seed, mean_bundle_size,
                                       standard_deviation_bundle_size)

    def get_efficient_allocation(self, display_output=False):
        """
        The efficient allocation is calculated on a generic definition. It is then "translated" into individual licenses that are assigned to bidders.
        Note that this does NOT result in a consistent allocation, since a single license can be assigned to multiple bidders.
        The value per bidder is still consistent, which is why this method can still be useful.
        """
        if self.efficient_allocation:
            return self.efficient_allocation, sum(
                [self.efficient_allocation[bidder_id]['value'] for bidder_id in self.efficient_allocation.keys()])

        mip = autoclass(self.mip_path)(self._bidder_list)
        mip.setDisplayOutput(display_output)

        allocation = mip.calculateAllocation()

        self.efficient_allocation = {}

        available = self.get_good_ids()

        for bidder_id, bidder in self.population.items():
            self.efficient_allocation[bidder_id] = {}
            self.efficient_allocation[bidder_id]['good_ids'] = []
            if allocation.getWinners().contains(bidder):
                bidder_allocation = allocation.allocationOf(bidder)
                bundle_entry_iterator = bidder_allocation.getBundle().getBundleEntries().iterator()
                while bundle_entry_iterator.hasNext():
                    bundle_entry = bundle_entry_iterator.next()
                    count = bundle_entry.getAmount()
                    licenses_iterator = cast(self.generic_definition_path,
                                             bundle_entry.getGood()).containedGoods().iterator()
                    given = 0
                    while licenses_iterator.hasNext() and given < count:
                        lic_id = licenses_iterator.next().getLongId()
                        if lic_id in available:
                            self.efficient_allocation[bidder_id]['good_ids'].append(lic_id)
                            available.remove(lic_id)
                            given += 1
                    assert given == count

            self.efficient_allocation[bidder_id][
                'value'] = bidder_allocation.getValue().doubleValue() if allocation.getWinners().contains(
                bidder) else 0.0

        return self.efficient_allocation, allocation.getTotalAllocationValue().doubleValue()
