from jnius import MetaJavaClass, JavaMethod

from .generic_model import GenericModel


class _Mrvm(GenericModel, metaclass=MetaJavaClass):
    # We have to define the java class and any method we're going to use in this child class
    __javaclass__ = 'org/spectrumauctions/sats/core/model/mrvm/MultiRegionModel'
    setNumberOfNationalBidders = JavaMethod('(I)V')
    setNumberOfRegionalBidders = JavaMethod('(I)V')
    setNumberOfLocalBidders = JavaMethod('(I)V')
    createWorld = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Lorg/spectrumauctions/sats/core/model/World;')
    createPopulation = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/model/World;Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Ljava/util/List;')

    def __init__(self, seed, number_of_national_bidders, number_of_regional_bidders, number_of_local_bidders):
        self.number_of_national_bidders = number_of_national_bidders
        self.number_of_regional_bidders = number_of_regional_bidders
        self.number_of_local_bidders = number_of_local_bidders
        super().__init__(
            seed=seed,
            mip_path='org.spectrumauctions.sats.opt.model.mrvm.MRVM_MIP',
            generic_definition_path='org.spectrumauctions.sats.core.model.mrvm.MRVMGenericDefinition'
        )

    def prepare_world(self):
        self.setNumberOfNationalBidders(self.number_of_national_bidders)
        self.setNumberOfRegionalBidders(self.number_of_regional_bidders)
        self.setNumberOfLocalBidders(self.number_of_local_bidders)

    def get_model_name(self):
        return ('MRVM')
