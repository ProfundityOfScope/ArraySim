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
from astropy.coordinates import ICRS, AltAz, SkyCoord, EarthLocation, Angle, ITRS
import astropy.units as u
from scipy.optimize import newton
import matplotlib.pyplot as plt
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

    def to_geocentric_at_time(self, time: Time):
        xyzt = np.zeros((3, len(self.instruments)))
        for i, instrument in enumerate(self.instruments):
            if isinstance(instrument, Station):
                xyzt[:, i] = instrument.to_geocentric()
            elif isinstance(instrument, Satellite):
                xyzt[:, i] = instrument.to_geocentric(time)
            else:
                raise ValueError
        return xyzt

    def to_geocentric_now(self):
        return self.to_geocentric_at_time(Time.now())


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

    def to_geocentric(self):
        return np.array(self.location.value.tolist())


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
        
        return np.squeeze(nu)
    
    def to_geocentric(self, time: Time):
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
        T = np.sqrt( a**3 / (6.6743e-11 * 5.972e24/(4*np.pi**2)) )
        return T
    
    def orbit_geocentric(self, n=50):
        period = self.get_period()
        time_array = self.epoch + np.linspace(0, 1, n) * period * u.s
        return self.to_geocentric(time_array)

    @classmethod
    def dummy(cls):
        return cls('Dummy', 7e6, 0.1, 30, 30, 30, 30, Time.now().iso)
    
    def __repr__(self):
        return f'Satellite(name={self.name}, semi_major={self.sma}, epoch={self.epoch})'


def xyz_to_lonlat(pos):
    rad = np.sqrt(np.sum(np.square(pos), axis=0))
    lon = np.arctan2(pos[1], pos[0])
    lat = np.arcsin(pos[2] / rad)
    
    return rad, lon, lat


if __name__=='__main__':
    testtime = Time.now()
    
    myarray = Array.from_file('hsa.antpos')
    testsat = Satellite('Fermi', 7e6, 0.001, 25.58, 29.29, 131.16, 229,
                        Time('2016-02-23 04:46:22'))
    myarray.instruments.append(testsat)
    
    sc = SkyCoord('00h02m00s +62d16\'00"', frame=ICRS)
    
    sites = myarray.instruments
    pos = myarray.to_geocentric_at_time(testtime) / 6.378e6
    maxr = np.max(np.sqrt(np.sum(np.square(pos), axis=0)))
    
    # Plot satellite orbit circle
    satcircle = testsat.orbit_geocentric()
    
    # Plot sky Circle drawn by celestial object
    skytime = testtime + np.linspace(0, 1, 50) * u.day
    skycircle = sc.transform_to(ITRS(obstime=skytime)).cartesian.xyz.value
    
    pr, plon, plat = xyz_to_lonlat(pos)
    sr, slon, slat = xyz_to_lonlat(satcircle)
    er, elon, elat = xyz_to_lonlat(skycircle)
    
    plt.figure(figsize=(10, 5), dpi=1920/10)
    plt.subplot(projection='mollweide')
    plt.scatter(plon, plat, label='Ground Site')
    plt.scatter(slon, slat, c='C1', s=5, label='Satellite Path')
    plt.scatter(elon, elat, c='C2', s=5)
    plt.scatter(elon[0], elat[0], ec='C2', fc='w', s=100, label='Celestial Object Position')
    plt.grid()
    plt.legend()
    plt.title(testtime.iso)
    plt.savefig('chaoticsurface.png')