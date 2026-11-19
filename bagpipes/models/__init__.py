from __future__ import print_function, division, absolute_import

from . import making

from .stellar_model import stellar
from .dust_emission_model import dust_emission
from .dust_attenuation_model import dust_attenuation
from .agn_model import agn

from .nebular_model import nebular
from .igm_model import igm

from .model_galaxy import model_galaxy
from .star_formation_history import star_formation_history
from .chemical_enrichment_history import chemical_enrichment_history

def list_available_components():
    """Return a dictionary of available component types and their classes."""
    return {
        'dust_emission': dust_emission.list_available_dust_emission_components(),
        'dust_attenuation': dust_attenuation.list_available_dust_attenuation_components(),
        'agn': agn.list_available_agn_components(),
        'nebular': nebular.list_available_nebular_components(),
        'star_formation_history': star_formation_history.list_available_sfh_components(),
        'chemical_enrichment_history': chemical_enrichment_history.list_available_ceh_components()
    }