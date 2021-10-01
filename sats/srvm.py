from jnius import MetaJavaClass, JavaMethod

from .generic_model import GenericModel


class _Srvm(GenericModel, metaclass=MetaJavaClass):
    # We have to define the java class and any method we're going to use in this child class
    __javaclass__ = 'org/spectrumauctions/sats/core/model/srvm/SingleRegionModel'
    setNumberOfSmallBidders = JavaMethod('(I)V')
    setNumberOfHighFrequencyBidders = JavaMethod('(I)V')
    setNumberOfSecondaryBidders = JavaMethod('(I)V')
    setNumberOfPrimaryBidders = JavaMethod('(I)V')
    createWorld = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Lorg/spectrumauctions/sats/core/model/srvm/SRVMWorld;')
    createPopulation = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/model/World;Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Ljava/util/List;')

    def __init__(self, seed, number_of_small_bidders, number_of_high_frequency_bidders, number_of_secondary_bidders,
                 number_of_primary_bidders):
        self.number_of_small_bidders = number_of_small_bidders
        self.number_of_high_frequency_bidders = number_of_high_frequency_bidders
        self.number_of_secondary_bidders = number_of_secondary_bidders
        self.number_of_primary_bidders = number_of_primary_bidders
        super().__init__(
            seed=seed,
            mip_path='org.spectrumauctions.sats.opt.model.srvm.SRVM_MIP',
            generic_definition_path='org.spectrumauctions.sats.core.model.srvm.SRVMBand'
        )

    def prepare_world(self):
        self.setNumberOfSmallBidders(self.number_of_small_bidders)
        self.setNumberOfHighFrequencyBidders(self.number_of_high_frequency_bidders)
        self.setNumberOfSecondaryBidders(self.number_of_secondary_bidders)
        self.setNumberOfPrimaryBidders(self.number_of_primary_bidders)

    def get_model_name(self):
        return ('SRVM')
