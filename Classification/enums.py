#!/usr/bin/env python

#   enums.py
#
#   Sam Heilbron
#   Last Updated: December 8, 2016
#
#   List of enums:
#		Color


def enum(**named_values):
	return type('Enum', (), named_values)

Color = enum(
	BLACK 	= (0, 0, 0),
	WHITE 	= (255, 255, 255),
	RED 	= (0, 0, 255),
	GREEN 	= (0, 255, 0),
	BLUE 	= (255, 0, 0),
	YELLOW	= (0, 255, 255))

Datatypes = enum(
	FLOAT64 = 'float64',
	INT		= 'int',
	INT16	= 'int16',
	UINT8	= 'uint8')