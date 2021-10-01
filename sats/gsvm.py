from jnius import MetaJavaClass, JavaMethod

from .simple_model import SimpleModel


class _Gsvm(SimpleModel, metaclass=MetaJavaClass):
    # We have to define the java class and any method we're going to use in this child class
    __javaclass__ = 'org/spectrumauctions/sats/core/model/gsvm/GlobalSynergyValueModel'
    setNumberOfNationalBidders = JavaMethod('(I)V')
    setNumberOfRegionalBidders = JavaMethod('(I)V')
    createWorld = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Lorg/spectrumauctions/sats/core/model/gsvm/GSVMWorld;')
    createPopulation = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/model/World;Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Ljava/util/List;')
    setLegacyGSVM = JavaMethod('(Z)V')

    def __init__(self, seed, number_of_national_bidders, number_of_regional_bidders, isLegacyGSVM=False):
        self.number_of_national_bidders = number_of_national_bidders
        self.number_of_regional_bidders = number_of_regional_bidders
        self.isLegacy = isLegacyGSVM
        super().__init__(
            seed=seed,
            mip_path='org.spectrumauctions.sats.opt.model.gsvm.GSVMStandardMIP'
        )

    def prepare_world(self):
        self.setNumberOfNationalBidders(self.number_of_national_bidders)
        self.setNumberOfRegionalBidders(self.number_of_regional_bidders)
        self.setLegacyGSVM(self.isLegacy)

    def get_model_name(self):
        return ('GSVM')
