
#A boundary condition class

class BoundaryCondition:

    def __init__(self, form, points_generator,weight,impose):

        self.form = form  # The function form of the boundary condition

        self.points_generator = points_generator  # Generate points on the boundary

        self.weight=weight # The weight of this boundary in calculating loss function
        
        self.impose=impose # Type int, whether this boundary condition is exactly imposed
                           #'1' represents exactly imposed and '0' represents no