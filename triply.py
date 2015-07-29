import googlemaps
import pandas as pd
from itertools import combinations
import numpy as np
import random

gmaps = googlemaps.Client(key="PASTE YOUR API KEY HERE")
def do_dis():
  df = pd.
class Itinerary:
    """Itinerary class takes a list of waypoints to build """
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.waypoint_distances = {}
        self.waypoint_durations = {}

    def get_distances(self):
        ##must pass waypoint_distances
        for (waypoint1, waypoint2) in combinations(self.waypoints, 2):
            try:
                route = gmaps.distance_matrix(origins=[waypoint1],
                                      destinations=[waypoint2],
                                      mode="driving", # Change this to "walking" for walking directions,
                                                      # "bicycling" for biking directions, etc.
                                      language="English",
                                      units="metric")

                                      # "distance" is in meters
                distance = route["rows"][0]["elements"][0]["distance"]["value"]

                # "duration" is in seconds
                duration = route["rows"][0]["elements"][0]["duration"]["value"]

                self.waypoint_distances[frozenset([waypoint1, waypoint2])] = distance
                self.waypoint_durations[frozenset([waypoint1, waypoint2])] = duration

            except Exception as e:
                print("Error with finding the route between %s and %s." % (waypoint1, waypoint2))

    def write_routes(self):
        ## should write to db
        with open("my-waypoints-dist-dur.tsv", "wb") as out_file:
            out_file.write("\t".join(["waypoint1",
                              "waypoint2",
                              "distance_m",
                              "duration_s"]))

            for (waypoint1, waypoint2) in self.waypoint_distances.keys():
                out_file.write("\n" +
                       "\t".join([waypoint1,
                                  waypoint2,
                                  str(self.waypoint_distances[frozenset([waypoint1, waypoint2])]),
                                  str(self.waypoint_durations[frozenset([waypoint1, waypoint2])])]))

    def get_distances_from_db():
        waypoint_dist = {}
        waypoint_dur = {}
        all_waypoints = set()

        waypoint_data = pd.read_csv("my-waypoints-dist-dur.tsv", sep="\t") # should read from db

        for i, row in waypoint_data.iterrows():
            waypoint_dist[frozenset([row.waypoint1, row.waypoint2])] = row.distance_m
            waypoint_dur[frozenset([row.waypoint1, row.waypoint2])] = row.duration_s
            all_waypoints.update([row.waypoint1, row.waypoint2])

        return waypoint_dist, waypoint_dur, all_waypoints
        #need to reset distances and class variables

    def compute_fitness(soln):
        solution_fitness = 0.0
        for index in range(len(solution)):
            waypoint1 = soln[index - 1]
            waypoint2 = soln[index]
            solution_fitness += self.waypoint_distances[frozenset([waypoint1, waypoint2])]

        return solution_fitness

    def generate_random_agent():
        new_random_agent = list(all_waypoints)
        random.shuffle(new_random_agent)
        return tuple(new_random_agent)

    def mutate_agent(agent_genome, max_mutations=3):
    """
        Applies 1 - `max_mutations` point mutations to the given road trip.

        A point mutation swaps the order of two waypoints in the road trip.
    """

        agent_genome = list(agent_genome)
        num_mutations = random.randint(1, max_mutations)

        for mutation in range(num_mutations):
            swap_index1 = random.randint(0, len(agent_genome) - 1)
            swap_index2 = swap_index1

            while swap_index1 == swap_index2:
                swap_index2 = random.randint(0, len(agent_genome) - 1)

            agent_genome[swap_index1], agent_genome[swap_index2] = agent_genome[swap_index2], agent_genome[swap_index1]

        return tuple(agent_genome)

    def shuffle_mutation(agent_genome):
    """
        Applies a single shuffle mutation to the given road trip.

        A shuffle mutation takes a random sub-section of the road trip
        and moves it to another location in the road trip.
    """

        agent_genome = list(agent_genome)

        start_index = random.randint(0, len(agent_genome) - 1)
        length = random.randint(2, 20)

        genome_subset = agent_genome[start_index:start_index + length]
        agent_genome = agent_genome[:start_index] + agent_genome[start_index + length:]

        insert_index = random.randint(0, len(agent_genome) + len(genome_subset) - 1)
        agent_genome = agent_genome[:insert_index] + genome_subset + agent_genome[insert_index:]

        return tuple(agent_genome)

    def generate_random_population(pop_size):
    """
        Generates a list with `pop_size` number of random road trips.
    """

        random_population = []
        for agent in range(pop_size):
            random_population.append(generate_random_agent())
        return random_population

    def run_genetic_algorithm(generations=5000, population_size=100):
    """
        The core of the Genetic Algorithm.
    """

        # Create a random population of `population_size` number of solutions.
        population = generate_random_population(population_size)

        # For `generations` number of repetitions...
        for generation in range(generations):

            # Compute the fitness of the entire current population
            population_fitness = {}

            for agent_genome in population:
                if agent_genome in population_fitness:
                    continue

                population_fitness[agent_genome] = compute_fitness(agent_genome)

                # Take the 10 shortest road trips and produce 10 offspring each from them
            new_population = []
            for rank, agent_genome in enumerate(sorted(population_fitness, key=population_fitness.get)[:10]):
                if (generation % 1000 == 0 or generation == generations - 1) and rank == 0:
                    print("Generation %d best: %d | Unique genomes: %d" % (generation,
                                                                       population_fitness[agent_genome],
                                                                       len(population_fitness)))
                    print(agent_genome)
                    print("")

                # Create 1 exact copy of each of the top 10 road trips
                new_population.append(agent_genome)

                # Create 2 offspring with 1-3 point mutations
                for offspring in range(2):
                    new_population.append(mutate_agent(agent_genome, 3))

                # Create 7 offspring with a single shuffle mutation
                for offspring in range(7):
                    new_population.append(shuffle_mutation(agent_genome))

                    # Replace the old population with the new population of offspring
            for i in range(len(population))[::-1]:
                del population[i]

            population = new_population
