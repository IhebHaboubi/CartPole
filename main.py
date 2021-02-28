import os
import pickle

import gym
import neat

env = gym.make('CartPole-v1')

env._max_episode_steps = 10000

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = 1 if net.activate(observation)[0] > 0.5 else 0 
            observation, reward, done, info = env.step(action)
            score += reward
        genome.fitness = score 


def run(config_file):
    
    if os.path.exists("checkpoint"):
        with open("checkpoint", "rb") as f:
            net = pickle.load(f)
    
    else:

        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        #p.add_reporter(neat.Checkpointer(5))

        # Run for up to 500 generations.
        winner = p.run(eval_genomes, 500)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        net = neat.nn.FeedForwardNetwork.create(winner, config)

        with open("checkpoint","wb") as f:
            pickle.dump(net, f)
           
    observation = env.reset()
    done = False
    while not done:
        env.render()
        action = 1 if net.activate(observation)[0] > 0.5 else 0 
        observation, reward, done, info = env.step(action)

run("config")
