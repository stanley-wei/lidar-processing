#########################
# CLASSIFICATION CONFIG #
#########################
class classes:

	# LAS Classification Codes
	UNASSIGNED = 1
	GROUND = 2
	LOW_VEGETATION = 3
	MED_VEGETATION = 4
	HIGH_VEGETATION = 5
	BUILDING = 6
	WATER = 9

	DALES_CLASSES = {
		1: GROUND, 
		2: HIGH_VEGETATION,
		3: UNASSIGNED,
		4: UNASSIGNED,
		5: UNASSIGNED,
		6: UNASSIGNED,
		7: UNASSIGNED,
		8: BUILDING
	}

	MY_CLASSES = {
		UNASSIGNED: 0,
		GROUND: 1,
		HIGH_VEGETATION: 2,
		BUILDING: 3
	}

	# ground(1), vegetation(2), cars(3), trucks(4), power lines(5), fences(6), poles(7) and buildings(8)