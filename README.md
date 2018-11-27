# Rl-Interface

<h2>Methods that agents need to implement:</h2>
  These methods are necessary for this library to work.
* pre_step(state)  
  A function that takes in the observation and returns an action to take.
  Gets called just before an action is taken

* post_step(self, state, action, reward, next_state, done)  
  A function that takes in the above parameters and returns nothing.
  This is for any processing that needs doing after an action is taken
  Gets called just after an action is taken

* post_episode()  
  This is a function that gets called after every episode, to handle post episode training and cleanup
