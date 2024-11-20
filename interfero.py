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


    def to_geocentric_at_time(self, time):
        xyzt = np.zeros((3, len(self.instruments)))
        for i, instrument in enumerate(self.instruments):
            if isinstance(instrument, Station):
                xyzt[:, i] = instrument.location.value.tolist()
            elif isinstance(instrument, Satellite):
                xyzt[:, i] = instrument.get_pos_at_time(time)
            else:
                raise ValueError
        return xyzt
    
    def to_geocentric_now(self):
        return self.to_geocentric_at_time(Time.now())
    
@dataclass
class Station:
    name: str
    lat: str
    lon: str
    elev: float
    diam: float
    
    def __post_init__(self):
        self._lat_ang = Angle(self.lat, unit='deg')
        self._lon_ang = Angle(self.lon, unit='deg')
        self.location = EarthLocation(lon=self._lon_ang,
                                      lat=self._lat_ang,
                                      height=self.elev)
    
@dataclass
class Satellite:
    name: str
    sma: float  # Semi-major axis (km)
    ecc: float  # ecc
    inc: float  # inc (degrees)
    raan: float  # Right Ascension of Ascending Node (degrees)
    aop: float  # Argument of Periapsis (degrees)
    mae: float  # Mean Anomaly at Epoch (degrees)
    epoch: Time  # Epoch for the orbital elements

    def get_pos_at_time(self, time: Time):
        """
        Compute geocentric position of satellite.

        Compute the geocentric (x, y, z) position of the satellite at a
        given time.
        """
        # Step 1: Compute mean motion
        mu = 6.6743e-11 * 5.972e24  # Gravitational parameter G * M (m^3/s^2)
        n = np.sqrt(mu / self.sma**3)  # Mean motion (rad/s)

        # Step 2: Compute mean anomaly at the given time
        delta_t = (time - self.epoch).to_value("sec")  # Time diff in seconds
        M = np.radians(self.mae) + n * delta_t
        M = np.mod(M, 2 * np.pi)  # Keep within 0 to 2pi

        # Step 3: Solve Kepler's equation for the eccentric anomaly
        def kepler_eq(E):
            return E - self.ecc * np.sin(E) - M

        E = newton(kepler_eq, M)  # Solve for eccentric anomaly (radians)

        # Step 4: Compute true anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + self.ecc) * np.sin(E / 2),
                            np.sqrt(1 - self.ecc) * np.cos(E / 2))

        # Step 5: Compute distance to the central body
        r = self.sma * (1 - self.ecc**2) / (1 + self.ecc * np.cos(nu))

        # Step 6: Satellite position in the orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = 0  # In orbital plane

        # Step 7: Rotate into geocentric equatorial frame
        i = np.radians(self.inc)
        Omega = np.radians(self.raan)
        omega = np.radians(self.aop)

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
        T = np.sqrt( self.sma**3 / (6.6743e-11 * 5.972e24/(4*np.pi**2)) )
        return T
    
def xyz_to_lonlat(pos):
    rad = np.sqrt(np.sum(np.square(pos), axis=0))
    lon = np.arctan2(pos[1], pos[0])
    lat = np.arcsin(pos[2] / rad)
    
    return rad, lon, lat
    
if __name__=='__main__':
    testtime = Time.now() + 1*u.hr
    
    myarray = Array.from_file('hsa.antpos')
    testsat = Satellite('Fermi', 7e6, 0.001, 25.58, 29.29, 131.16, 229,
                        Time('2016-02-23 04:46:22'))
    myarray.instruments.append(testsat)
    
    sc = SkyCoord('00h02m00s +62d16\'00"', frame=ICRS)
    sc_v = sc.transform_to(ITRS(obstime=testtime))
    vec = sc_v.cartesian.xyz.value
    
    sites = myarray.instruments
    pos = myarray.to_geocentric_at_time(testtime) / 6.378e6
    maxr = np.max(np.sqrt(np.sum(np.square(pos), axis=0)))
    
    # Plot satellite orbit circle
    sattime = testtime + np.linspace(0, 1, 50) * testsat.get_period() * u.s
    satcircle = np.array([ testsat.get_pos_at_time(t)/6.378e6 for t in sattime ]).T
    
    # Plot sky Circle drawn by celestial object
    skytime = testtime + np.linspace(0, 1, 50) * u.day
    skycircle = np.array([ sc.transform_to(ITRS(obstime=t)).cartesian.xyz.value for t in skytime ]).T
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection='3d')
    ax.scatter(*pos)
    ax.set_xlim(-maxr, maxr)
    ax.set_ylim(-maxr, maxr)
    ax.set_zlim(-maxr, maxr)
    ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]])
    ax.plot(*satcircle)
    ax.plot(*skycircle)
    plt.show()
    
    pr, plon, plat = xyz_to_lonlat(pos)
    sr, slon, slat = xyz_to_lonlat(satcircle)
    er, elon, elat = xyz_to_lonlat(skycircle)
    
    plt.figure(figsize=(10, 5), dpi=1920/10)
    plt.subplot(projection='mollweide')
    plt.scatter(plon, plat, label='Ground Site')
    plt.scatter(slon, slat, c='C1', s=5)
    plt.scatter(slon[0], slat[0], ec='C1', fc='w', s=100, label='Satellite Position')
    plt.scatter(elon, elat, c='C2', s=5)
    plt.scatter(elon[0], elat[0], ec='C2', fc='w', s=100, label='Celestial Object Position')
    plt.grid()
    plt.legend()
    plt.title(testtime.iso)
    plt.savefig('chaoticsurface.png')