# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 00:07:22 2020

@author: mrgoh
"""

import pandas as pd

LDAP = pd.read_csv("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\LDAP.csv")
LDAP.shape
LDAP.columns = LDAP.columns.str.strip()
LDAP['Label'].value_counts()

MSSQL = pd.read_csv("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\MSSQL.csv")
MSSQL.shape
MSSQL.columns = MSSQL.columns.str.strip()
MSSQL['Label'].value_counts()

NetBIOS = pd.read_csv("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\NetBIOS.csv")
NetBIOS.shape
NetBIOS.columns = NetBIOS.columns.str.strip()
NetBIOS['Label'].value_counts()

Syn = pd.read_csv("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\Syn.csv")
Syn.shape
Syn.columns = Syn.columns.str.strip()
Syn['Label'].value_counts()

UDP = pd.read_csv("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\UDP.csv")
UDP.shape
UDP.columns = UDP.columns.str.strip()
UDP['Label'].value_counts()

UDPLag = pd.read_csv("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\UDPLag.csv")
UDPLag.shape
UDPLag.columns = UDPLag.columns.str.strip()
UDPLag['Label'].value_counts()

Portmap = pd.read_csv("E:\\MS SEM2\\CIS 694\\Term Paper\\CICDDoS2019\\Portmap.csv")
Portmap.shape
Portmap.columns = Portmap.columns.str.strip()
Portmap['Label'].value_counts()
