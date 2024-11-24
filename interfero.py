#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:18:09 2024

@author: bruzewskis
"""

import numpy as np
from dataclasses import dataclass
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import ICRS, ITRS, GCRS, AltAz, SkyCoord, EarthLocation, Angle
import astropy.units as u
from scipy.optimize import newton
from typing import Union


class Array:
    def __init__(self, instruments: list):
        self.instruments = instruments

    @classmethod
    def from_file(cls, filename: str):
        dat = Table.read(filename, delimiter=';', format='csv')
        stations = []
        for i in range(len(dat)):
            stations.append(Station(name=dat['Code'][i],
                                    lat=dat['NorthLatitude'][i],
                                    lon=dat['EastLongitude'][i],
                                    elev=dat['Elevation'][i],
                                    diam=dat['Diameter'][i]))
        return cls(instruments=stations)

    def to_gcrs(self, time: Time):
        time = np.atleast_1d(time)
        xyzt = np.zeros((3, len(self.instruments), len(time)))
        for i, instrument in enumerate(self.instruments):
            xyzt[:, i] = instrument.to_gcrs(time)
        return xyzt

    def to_gcrs_now(self):
        return self.to_gcrs(Time.now())
    
    def to_projection(self, time: Time, sc: SkyCoord):
        time = np.atleast_1d(time)
        
        lon = -sc.ra.rad
        lat = sc.dec.rad
        
        Ry = np.array([[np.cos(lat), 0, np.sin(lat)],
                       [0, 1, 0],
                       [-np.sin(lat), 0, np.cos(lat)]])  # about y by lat/dec
        Rz = np.array([[np.cos(lon), -np.sin(lon), 0],
                       [np.sin(lon), np.cos(lon), 0],
                       [0, 0, 1]])  # about z by lon/ra
        R = Ry @ Rz
        
        positions = self.to_gcrs(time)
        rot_pos = np.einsum('ij,jst->ist', R, positions)
        
        return rot_pos
        
    
    def to_uvw(self, time: Time, sc: SkyCoord, only_valid_baselines=False):
        time = np.atleast_1d(time)
        
        rot_pos = self.to_projection(time, sc)
        print('rotpos', rot_pos.shape)
        
        earth_rad = 6.378e6
        elevation_limit = np.deg2rad(00) # HARD SET
        proj_rad = np.sqrt(rot_pos[1]**2 + rot_pos[2]**2)
        
        # Satellite visible or not
        issat = np.array([isinstance(s, Satellite) for s in self.instruments])
        in_earth_shadow = np.logical_and(rot_pos[0]<0, proj_rad<earth_rad)
        cond1 = issat[:, None] & ~in_earth_shadow
        
        # Ground station visible or not (elevation)
        above_elev_lim = rot_pos[0] > np.sin(elevation_limit) * earth_rad
        cond2 = ~issat[:, None] & above_elev_lim
        has_sightline = cond1 | cond2
        
        # Calculate all baselines
        uvw = rot_pos[:, None, :] - rot_pos[:, :, None]
        
        if only_valid_baselines:
            valid_baseline = has_sightline[None, :] & has_sightline[:, None]
            uvw[:, ~valid_baseline] = np.nan

        return rot_pos, has_sightline, uvw


class AngleProperty:
    def __init__(self, name):
        self.name = f'_{name}'

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        if not isinstance(value, Angle):
            value = Angle(value, unit=u.deg)
        setattr(instance, self.name, value)

class DateProperty:
    def __init__(self, name):
        self.name = f'_{name}'

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        if not isinstance(value, Time):
            value = Time(value)
        setattr(instance, self.name, value)


class Station:

    lat = AngleProperty('lat')
    lon = AngleProperty('lon')

    def __init__(self, name: str, lat: Union[float, str, Angle, u.Quantity],
                 lon: Union[float, str, Angle, u.Quantity],
                 elev: float, diam: float):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.diam = diam
        self.location = EarthLocation(lon=self.lon,
                                      lat=self.lat,
                                      height=self.elev)

    def to_geocentric(self, time: Time=None):
        return np.array(self.location.value.tolist())
    
    def to_gcrs(self, time: Time):
        loc_itrs = self.location.get_itrs(obstime=time)
        loc_gcrs = loc_itrs.transform_to(GCRS(obstime=time))
        return loc_gcrs.cartesian.xyz.value
    
    def __repr__(self):
        return f'Station(name={self.name})'
        
    
    @classmethod
    def dummy(cls):
        return cls('Dummy', 45, 45, 10, 10)


class Satellite:
    inclination = AngleProperty('inclination')
    ra_asc = AngleProperty('ra_asc')
    arg_of_peri = AngleProperty('arg_of_peri')
    mean_anom = AngleProperty('mean_anom')
    epoch = DateProperty('epoch')

    def __init__(self, name: str, semi_major: float, eccen: float,
                 inclination: Union[float, str, Angle, u.Quantity],
                 ra_asc: Union[float, str, Angle, u.Quantity],
                 arg_of_peri: Union[float, str, Angle, u.Quantity],
                 mean_anom: Union[float, str, Angle, u.Quantity],
                 epoch: Time):
        self.name = name
        self.semi_major = semi_major
        self.eccen = eccen
        self.inclination = inclination
        self.ra_asc = ra_asc
        self.arg_of_peri = arg_of_peri
        self.mean_anom = mean_anom
        self.epoch = epoch

    def true_anomaly(self, time: Time) -> np.ndarray:

        # Validate time
        time = np.atleast_1d(time)

        # Step 1: Compute mean motion
        mu = 6.6743e-11 * 5.972e24  # Gravitational parameter G * M (m^3/s^2)
        n = np.sqrt(mu / self.semi_major**3)  # Mean motion (rad/s)

        # Step 2: Compute mean anomaly at the given time
        delta_t = (time - self.epoch).to_value("sec")  # Time diff in seconds
        M = self.mean_anom.rad + n * delta_t
        M = np.mod(M, 2 * np.pi)  # Keep within 0 to 2pi

        # Step 3: Solve Kepler's equation for the eccentric anomaly
        def kepler_eq(E):
            return E - self.eccen * np.sin(E) - M

        E = newton(kepler_eq, M)  # Solve for eccentric anomaly (radians)

        # Step 4: Compute true anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + self.eccen) * np.sin(E / 2),
                            np.sqrt(1 - self.eccen) * np.cos(E / 2))

        return nu

    def to_gcrs(self, time: Time):
        '''
        This assumes that the orbital elements are all you need, in reality 
        this will need to get the TLEs and propogate them forward

        '''
        nu = self.true_anomaly(time)

        # Step 5: Compute distance to the central body
        r = self.semi_major * (1 - self.eccen**2) / (1 + self.eccen * np.cos(nu))

        # Step 6: Satellite position in the orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = np.full_like(x_orb, 0)  # In orbital plane

        # Step 7: Rotate into geocentric equatorial frame
        i = self.inclination.rad
        Omega = self.ra_asc.rad
        omega = self.arg_of_peri.rad

        # Rotation matrices
        R_z_Omega = np.array([[np.cos(Omega), -np.sin(Omega), 0],
                              [np.sin(Omega),  np.cos(Omega), 0],
                              [0, 0, 1]])
        R_x_i = np.array([[1, 0, 0],
                          [0, np.cos(i), -np.sin(i)],
                          [0, np.sin(i),  np.cos(i)]])
        R_z_omega = np.array([[np.cos(omega), -np.sin(omega), 0],
                              [np.sin(omega),  np.cos(omega), 0],
                              [0, 0, 1]])

        # Total rotation matrix
        R = R_z_Omega @ R_x_i @ R_z_omega

        # Position vector in orbital plane
        pos_orb = np.array([x_orb, y_orb, z_orb])

        # Rotate to geocentric coordinates
        pos_geocentric = R @ pos_orb

        return pos_geocentric

    def get_period(self):
        a = self.semi_major
        T = np.sqrt(a**3 / (6.6743e-11 * 5.972e24/(4*np.pi**2)))
        return T

    def full_orbit(self, n=50):
        period = self.get_period()
        time_array = self.epoch + np.linspace(0, 1, n) * period * u.s
        return self.to_gcrs(time_array)

    @classmethod
    def dummy(cls):
        return cls('Dummy', 7e6, 0.1, 30, 30, 30, 30, Time.now().iso)

    def __repr__(self):
        return f'Satellite(name={self.name})'

def xyz_to_lonlat(pos):
    rad = np.sqrt(np.sum(np.square(pos), axis=0))
    lon = np.arctan2(pos[1], pos[0])
    lat = np.arcsin(pos[2] / rad)

    return rad, lon, lat


if __name__=='__main__':
    # Decide on time
    n_steps = 100
    time_start = Time.now()
    time_length = 3 * u.hour
    time_arr = time_start + np.linspace(-0.5, 0.5, n_steps) * time_length
    
    # consts
    r_earth = 6.378e6
    
    # Decide on target (zenith)
    location_dc = EarthLocation(lat=38.9215, lon=-77.0669, height=0)
    altaz_frame = AltAz(obstime=Time.now(), location=location_dc)
    zenith_altaz = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=altaz_frame)
    sc = zenith_altaz.transform_to('icrs')

    # Set up array from HSA, add Fermi satellite
    myarray = Array.from_file('hsa.antpos')
    myarray.instruments.append(Satellite('HALCA',
                                         1.7259e7, 0.599, 31.2,
                                         128, 144, 358, 
                                         Time('2016-04-28 09:56:22')))
    
    # Run the actual simulation
    pos_over_time, isvis2d, uvw2d = myarray.to_uvw(time_arr, sc, True)
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from tqdm import tqdm
    
    # Some useful scales and values for plotting
    pos_over_time /= r_earth
    uvw2d /= r_earth
    max_xy = 1.05*np.max(np.abs(pos_over_time[1:]))
    max_z = 1.05*np.max(np.abs(pos_over_time[0]))
    max_uv = 1.05*np.nanmax(np.abs(uvw2d[1:]))
    max_w = 1.05*np.nanmax(np.abs(uvw2d[0]))
    
    # First figure shows all relevant sightlines
    plt.figure(figsize=(21,10), dpi=1920/10)
    ax = plt.subplot2grid((1, 2), (0, 0), aspect='equal')
    for i in range(len(myarray.instruments)):
        vis = isvis2d[i]
        ax.scatter(pos_over_time[1, i, vis], 
                    pos_over_time[2, i, vis], 
                    c=pos_over_time[0, i, vis], cmap='Spectral',
                    vmin=-max_z, vmax=max_z, zorder=10)
        
    ax.set_xlim(-max_xy, max_xy)
    ax.set_ylim(-max_xy, max_xy)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Sightlines')
    ax.add_artist(Circle((0,0), 1, fc='none', ec='k'))
    # plt.axis('off')
    
    ax = plt.subplot2grid((1, 2), (0, 1), aspect='equal')
    ax.scatter(uvw2d[1].ravel(), uvw2d[2].ravel(), c=uvw2d[0].ravel(),
               cmap='Spectral', vmin=-max_w, vmax=max_w)
    ax.set_xlim(-max_uv, max_uv)
    ax.set_ylim(-max_uv, max_uv)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Baselines')
    
    plt.tight_layout()
    plt.show()
    
    # plt.figure(figsize=(10,10))
    # ax = plt.subplot(projection='3d')
    # ax.scatter(uvw2d[0].ravel(), uvw2d[1].ravel(), uvw2d[2].ravel(), marker='.')
    
    # ax.set_xlim(-2*max_xy, 2*max_xy)
    # ax.set_ylim(-2*max_xy, 2*max_xy)
    # ax.set_zlim(-2*max_xy, 2*max_xy)