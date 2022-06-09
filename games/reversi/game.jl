import AlphaZero.GI

using Crayons
using StaticArrays

const BOARD_SIZE = 8
const NUM_CELLS = BOARD_SIZE ^ 2
# const TO_CONNECT = 4

const Player = UInt8
const WHITE = 0x01
const BLACK = 0x02

other(p::Player) = 0x03 - p

const Cell = UInt8
const EMPTY = 0x00
const Board = SVector{NUM_CELLS, Cell}
#const Board = SMatrix{NUM_COLS, NUM_ROWS, Cell, NUM_CELLS}

const INITIAL_BOARD = Board(zeros(NUM_CELLS))
#const INITIAL_BOARD = @SMatrix zeros(Cell, NUM_COLS, NUM_ROWS)
const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=BLACK)

## So, basically, this is the PARAMETERS for the game.
## For example, if you want to parameterize grid size, you could.
struct GameSpec <: GI.AbstractGameSpec end


## The GameEnv holds a game spec and a current state.
## Basically the "environment" for the AI
mutable struct GameEnv <: GI.AbstractGameEnv
  board :: Board
  curplayer :: Player
  initializing :: Bool
  finished :: Bool
  winner :: Player
end


## Returns the game specification of an environment
# Nothing in the spec, can just return empty spec.
GI.spec(::GameEnv) = GameSpec()


## New game environment in an inital state (possibly random)
function GI.init(::GameSpec)
  board = INITIAL_STATE.board
  curplayer = INITIAL_STATE.curplayer
  initializing = true
  finished = false
  winner = 0x00
  return GameEnv(board, curplayer, initializing, finished, winner)
end


## Modify the state of the GameEnv with `state`
function GI.set_state!(g::GameEnv, state)
  g.board = state.board
  g.curplayer = state.curplayer
  update_status!(g)
end


## Returns whether or not the game is a two-player game
GI.two_players(::GameSpec) = true


# 0 is the pass action, available only when all other actions are unavailable.
# 1-64 represent the corresponding cells on the board.
const ACTIONS = collect(0:NUM_CELLS)
const NUM_ACTIONS = length(ACTIONS)


## A vector of *all* actions (available or not)
GI.actions(::GameSpec) = ACTIONS


## Returns an independent copy of the environment.
function GI.clone(g::GameEnv)
  GameEnv(g.board, g.curplayer, g.initializing, g.finished, g.winner)
end

history(g::GameEnv) = g.history

#####
##### Defining game rules
#####

const Position = Tuple{UInt8, UInt8}

xy_to_pos(x::UInt8, y::UInt8) = (y - 1) * BOARD_SIZE + x
pos_to_xy(pos::UInt8) = (ceil(pos / BOARD_SIZE), ((pos - 1) % BOARD_SIZE) + 1)

# `player` is either
# EMPTY, WHITE, BLACK
function isValidMove(b::Board, player::UInt8, pos::Position) end
function updateOnPlay!(b::Board, player::UInt8, pos::Position) end


## Boolean vector for which actions are available
## The following must hold:
##    - game_terminated(game) || any(actions_mask(game))
##    - length(actions_mask(game)) == length(actions(spec(game)))
function GI.actions_mask(g::GameEnv)
  if (g.initializing)
    initial_actions_mask(g)
  else
    normal_actions_mask(g)
  end
end

function initial_actions_mask(g::GameEnv)
  mask = MVector{1 + NUM_CELLS, Bool}(repeat([false], NUM_CELLS))
  center::UInt8 = BOARD_SIZE ÷ 2

  allow_action_if_empty(mask, g.board, xy_to_pos(center, center))
  allow_action_if_empty(mask, g.board, xy_to_pos(center, center + 1))
  allow_action_if_empty(mask, g.board, xy_to_pos(center + 1, center))
  allow_action_if_empty(mask, g.board, xy_to_pos(center + 1, center + 1))
end

function allow_action_if_empty(mask::MVector{NUM_ACTIONS, Bool}, board::Board, pos::UInt8)
  if (board[pos] == EMPTY)
    mask[pos + 1] = true
  end
end

function normal_actions_mask(g::GameEnv)
  mask = MVector{1 + NUM_CELLS, Bool}(repeat([false], NUM_CELLS))
  has_valid_move::Bool = false
  
  for testPos = 1:NUM_CELLS
    xy::Position = pos_to_xy(testPos)
    if isValidMove(g.board, g.curplayer, xy)
      has_valid_move = mask[testPos + 1] = true
    end
  end

  if has_valid_move
    mask[1] = true
  end
  
  return mask
end

valid_pos((col, row)) = 1 <= col <= NUM_COLS && 1 <= row <= NUM_ROWS


count_pieces(b::Board, p::Player) = count(c -> c == p, b)


# Update statuses for `finished` and `winner`
initializing_board(b::Board) = count_pieces(b, EMPTY) > NUM_CELLS - 4

function update_status!(g::GameEnv)
  if (g.initializing)
    g.initializing = initializing_board(g.board)
    g.finished = false
    g.winner = EMPTY
  else
    g.finished = any(1:64) do pos
      g.board[pos] != EMPTY || 
        isValidMove(g.board, WHITE, pos) ||
        isValidMove(g.board, BLACK, pos)
    end

    if (g.finished)
      white_count = count_pieces(g.board, WHITE)
      black_count = count_pieces(g.board, BLACK)

      if (black_count > white_count)
        g.winner = BLACK
      elseif (black_count < white_count)
        g.winner = WHITE
      else
        g.winner = EMPTY
      end
    end
end


## Update game environment with chosen action (for current player).
function GI.play!(g::GameEnv, action)
  if action != 0  # 'pass' action
    g.board = updateOnPlay!(g.board, g.curplayer, action)
    g.curplayer = other(g.curplayer)
    update_status!(g)
  end
end


## Gets the current state of the game.
## WARNING: Result must not change.
GI.current_state(g::GameEnv) = (board=g.board, curplayer=g.curplayer)


## Returns `true` if it is white's turn, `false` otherwise.
GI.white_playing(g::GameEnv) = g.curplayer == WHITE

#####
##### Reward shaping
#####


## Boolean for whether or not the game is over.
function GI.game_terminated(g::GameEnv)
  return g.finished
end


## Returns the *immediate* reward obtained by the white player after last transition.
# TODO: Consider better reward functions.
function GI.white_reward(g::GameEnv)
  if g.finished
    g.winner == WHITE && (return  1.)
    g.winner == BLACK && (return -1.)
    return 0.
  else
    return 0.
  end
end

#####
##### Simple heuristic for minmax
#####


#=
const Pos = Tuple{Int, Int}
const Alignment = Vector{Pos}

function alignment_from(pos, dir) :: Union{Alignment, Nothing}
  al = Alignment()
  for i in 1:TO_CONNECT
    valid_pos(pos) || (return nothing)
    push!(al, pos)
    pos = pos .+ dir
  end
  return al
end

function alignments_with(dir) :: Vector{Alignment}
  als = [alignment_from((x, y), dir) for x in 1:NUM_COLS for y in 1:NUM_ROWS]
  return filter(al -> !isnothing(al), als)
end

const ALIGNMENTS = [
  alignments_with((1,  1));
  alignments_with((1, -1));
  alignments_with((0,  1));
  alignments_with((1,  0))]

function alignment_value_for(g::GameEnv, player, alignment)
  γ = 0.1
  N = 0
  for pos in alignment
    cell = g.board[pos...]
    if cell == player
      N += 1
    elseif cell == other(player)
      return 0.
    end
  end
  return γ ^ (TO_CONNECT - 1 - N)
end

function heuristic_value_for(g::GameEnv, player)
  return sum(alignment_value_for(g, player, al) for al in ALIGNMENTS)
end


## Heuristic estimate of the state for *current player*.
## Not needed for AlphaZero, but is useful for baselines like minimax.
function GI.heuristic_value(g::GameEnv)
  mine = heuristic_value_for(g, g.curplayer)
  yours = heuristic_value_for(g, other(g.curplayer))
  return mine - yours
end
=#


#####
##### ML interface
#####

flip_cell_color(c::Cell) = c == EMPTY ? EMPTY : other(c)

function flip_colors(board)
  return @SMatrix [
    flip_cell_color(board[pos])
    for col in 1:NUM_CELLS]
end


## Returns the vectorized version of a gamestate.
## Basically, an array of floats given to the neural network (I think)
function GI.vectorize_state(::GameSpec, state)
  board = state.curplayer == WHITE ? state.board : flip_colors(state.board)
  return Float32[
    board[pos] == c
    for pos in 1:NUM_CELLS,
        c in [EMPTY, WHITE, BLACK]]
end

#####
##### Symmetries
#####

function generate_dihedral_symmetries()
  N = BOARD_SIDE
  rot((x, y)) = (y, N - x + 1) # 90° rotation
  flip((x, y)) = (x, N - y + 1) # flip along vertical axis
  ap(f) = p -> pos_of_xy(f(xy_of_pos(p)))
  sym(f) = map(ap(f), collect(1:NUM_CELLS))
  rot2 = rot ∘ rot
  rot3 = rot2 ∘ rot
  return [
    sym(rot), sym(rot2), sym(rot3),
    sym(flip), sym(flip ∘ rot), sym(flip ∘ rot2), sym(flip ∘ rot3)]
end

const SYMMETRIES = generate_dihedral_symmetries()


## Returns a vector of gamestate symmetries.
function GI.symmetries(::GameSpec, s)
  return [
    ((board=Board(s.board[sym]), curplayer=s.curplayer), sym)
    for sym in SYMMETRIES]
end

#####
##### User interface
#####

col_letter(number) = 'A' + number - 1

## Returns a string representing a given action.
function GI.action_string(::GameSpec, a)
  if (a == 0)
    "$a) pass"
  else
    xy = pos_to_xy(a)
    "$a) Play tile at $(col_letter(xy[0]))$(xy[1])"
  end
end


## Returns the action denoted by `str`
function GI.parse_action(g::GameSpec, str)
  try
    p = parse(Int, str)
    0 <= p <= NUM_CELLS ? p : nothing
  catch
    nothing
  end
end

# 1 2 3 4 5 6 7
# . . . . . . .
# . . . . . . .
# . . . . . . .
# . . o x . . .
# . o o o . . .
# o x x x . x .

player_color(p) = p == WHITE ? crayon"light_red" : crayon"light_blue"
player_name(p)  = p == WHITE ? "Red (W)" : "Blue (B)"
player_mark(p)  = p == WHITE ? "O" : "X"
cell_mark(c)    = c == EMPTY ? "." : player_mark(c)
cell_color(c)   = c == EMPTY ? crayon"" : player_color(c)

## Prints the current game state.
function GI.render(g::GameEnv; with_position_names=true, botmargin=true)
  pname = player_name(g.curplayer)
  pcol = player_color(g.curplayer)
  print(pcol, pname, " plays:", crayon"reset", "\n\n")
  
  # Print legend
  print("  ")
  for y in 1:BOARD_SIZE
    print(col_letter(y), " ")
  end
  print("\n")

  # Print board
  for row in BOARD_SIZE:-1:1
    print(row, " ")
    for col in 1:NUM_COLS
      pos = xy_to_pos(col, row)
      c = g.board[pos]
      print(cell_color(c), cell_mark(c), crayon"reset", " ")
    end
    print("\n")
  end

  botmargin && print("\n")
end

#=
## Reads a state from standard input. (")
## Honestly, I don't know what this is supposed to do.
function GI.read_state(::GameSpec)
  board = Array(INITIAL_BOARD)
  try
    for col in 1:NUM_COLS
      input = readline()
      for (row, c) in enumerate(input)
        c = lowercase(c)
        if c ∈ ['o', 'w', '1']
          board[col, row] = WHITE
        elseif c ∈ ['x', 'b', '2']
          board[col, row] = BLACK
        end
      end
    end
    nw = count(==(WHITE), board)
    nb = count(==(BLACK), board)
    if nw == nb
      curplayer = WHITE
    elseif nw == nb + 1
      curplayer = BLACK
    else
      return nothing
    end
    return (board=board, curplayer=curplayer)
  catch e
    return nothing
  end
end
=#
