import os


class PySats:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if PySats.__instance == None:
            PySats()
        return PySats.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if PySats.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            import jnius_config
            # If you have troubles with the CLASSPATH, use the absolute path below:
            # jnius_config.set_classpath('C://Users//*')
            jnius_config.set_classpath('.', os.path.join('lib', '*'))
            PySats.__instance = self

    def create_lsvm(self, seed=None, number_of_national_bidders=1, number_of_regional_bidders=5, isLegacyLSVM=False):
        from sats.lsvm import _Lsvm
        return _Lsvm(seed, number_of_national_bidders, number_of_regional_bidders, isLegacyLSVM)

    def create_gsvm(self, seed=None, number_of_national_bidders=1, number_of_regional_bidders=6, isLegacyGSVM=False):
        from sats.gsvm import _Gsvm
        return _Gsvm(seed, number_of_national_bidders, number_of_regional_bidders, isLegacyGSVM)

    def create_mrvm(self, seed=None, number_of_national_bidders=3, number_of_regional_bidders=4,
                    number_of_local_bidders=3):
        from sats.mrvm import _Mrvm
        return _Mrvm(seed, number_of_national_bidders, number_of_regional_bidders, number_of_local_bidders)

    def create_srvm(self, seed=None, number_of_small_bidders=2, number_of_high_frequency_bidders=1,
                    number_of_secondary_bidders=2, number_of_primary_bidders=2):
        from sats.srvm import _Srvm
        return _Srvm(seed, number_of_small_bidders, number_of_high_frequency_bidders,
                     number_of_secondary_bidders, number_of_primary_bidders)
