import jax.numpy as jnp

from jaxcolors import utils


def spectrum_to_XYZ(wavelengths, values, str_color_space="cie1931", str_illuminant="d65"):
    assert isinstance(wavelengths, jnp.ndarray)
    assert isinstance(values, jnp.ndarray)
    assert wavelengths.ndim == 1
    assert values.ndim == 1
    assert wavelengths.shape[0] == values.shape[0]

    cmfs = utils.get_cmfs(wavelengths, str_color_space=str_color_space)
    illuminant = utils.get_illuminant(wavelengths, str_illuminant=str_illuminant)

    assert cmfs.ndim == 2
    assert illuminant.ndim == 2
    assert wavelengths.shape[0] == cmfs.shape[0] == illuminant.shape[0]
    assert cmfs.shape[1] == 4
    assert illuminant.shape[1] == 2
    assert jnp.all(cmfs[:, 0] == wavelengths)
    assert jnp.all(wavelengths == cmfs[:, 0])
    assert jnp.all(wavelengths == illuminant[:, 0])

    x = cmfs[:, 1]
    y = cmfs[:, 2]
    z = cmfs[:, 3]
    I = illuminant[:, 1]

    delta_wavelengths = wavelengths[1:] - wavelengths[:-1]
    Iy = I * y
    SIx = values * I * x
    SIy = values * I * y
    SIz = values * I * z

    denominator = jnp.sum(((Iy[1:] + Iy[:-1]) / 2) * delta_wavelengths)

    numerator_X = jnp.sum(((SIx[1:] + SIx[:-1]) / 2) * delta_wavelengths)
    numerator_Y = jnp.sum(((SIy[1:] + SIy[:-1]) / 2) * delta_wavelengths)
    numerator_Z = jnp.sum(((SIz[1:] + SIz[:-1]) / 2) * delta_wavelengths)

    X = numerator_X / denominator
    Y = numerator_Y / denominator
    Z = numerator_Z / denominator

    return jnp.array([X, Y, Z])
