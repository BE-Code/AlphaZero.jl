import Base.+
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
const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=WHITE)

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

const Position = Tuple{Int8, Int8}
toPos(p)::Position = (Int8(p[1]), Int8(p[2]))
+(a::Position, b::Position)::Position = (a[1] + b[1], a[2] + b[2])

pos_to_index(xy::Position)::UInt8 = (xy[2] - 1) * BOARD_SIZE + xy[1]
index_to_pos(pos::UInt8)::Position = (ceil(pos / BOARD_SIZE), ((pos - 1) % BOARD_SIZE) + 1)

const DIRECTIONS = [toPos(pos) for pos in [(-1, -1), (0, -1), (1, -1),
                                           (-1,  0),          (1,  0),
                                           (-1,  1), (0,  1), (1,  1)]]

# `player` is either
# EMPTY, WHITE, BLACK
function is_valid_move(b::Board, player::Player, pos::Position)
  opponent = if (player == WHITE) BLACK else WHITE end
  posXY = index_to_pos(pos)

  if b[pos] == EMPTY
    # loop through eight directions
    for direction in DIRECTIONS
      if is_valid_move_direction(b, player, posXY, direction)
        return true
      end
    end
  end
  return false
end

function is_valid_move_direction(b::Board, player::Player, pos::Position, direction::Position)
  opponent = if (player == WHITE) BLACK else WHITE end
  next = pos + direction
  while b[pos_to_index(next)] == opponent
    next += position
    if !valid_pos(next)
      return false
    elseif b[pos_to_index(next)] == player
      return true
    end
  end

  return false
end

function update_game_board_move(b::Board, player::Player, pos::Position)
  newBoard = MVector(b)
  opponent = if (player == WHITE) BLACK else WHITE end

  # loop through eight directions
  for direction in DIRECTIONS
    if is_valid_move_direction(b, player, pos, direction)
      next = pos + direction
      while b[pos_to_index(next)] != player
        newBoard[pos_to_index(next)] = player
        next += direction
      end
    end
  end

  return SVector(newBoard)
end


## Boolean vector for which actions are available
## The following must hold:
##    - game_terminated(game) || any(actions_mask(game))
##    - length(actions_mask(game)) == length(actions(spec(game)))
function GI.actions_mask(g::GameEnv)
  println("actions_mask")
  if (g.initializing)
    return initial_actions_mask(g)
  else
    return normal_actions_mask(g)
  end
end

function allow_action_if_empty(mask::Vector{Bool}, board::Board, pos::UInt8)
  if (board[pos] == EMPTY)
    mask[pos + 1] = true
  end
end

function initial_actions_mask(g::GameEnv)
  mask = Vector{Bool}(repeat([false], NUM_ACTIONS))
  center_val::UInt8 = BOARD_SIZE ÷ 2
  center::Position = (center_val, center_val)

  allow_action_if_empty(mask, g.board, pos_to_index(center + toPos((0, 0))))
  allow_action_if_empty(mask, g.board, pos_to_index(center + toPos((0, 1))))
  allow_action_if_empty(mask, g.board, pos_to_index(center + toPos((1, 0))))
  allow_action_if_empty(mask, g.board, pos_to_index(center + toPos((1, 1))))

  return mask
end

function normal_actions_mask(g::GameEnv)
  mask = Vector{Bool}(repeat([false], NUM_ACTIONS))
  has_valid_move::Bool = false
  
  for testPos = 1:NUM_CELLS
    xy::Position = index_to_pos(testPos)
    if is_valid_move(g.board, g.curplayer, xy)
      has_valid_move = mask[testPos + 1] = true
    end
  end

  if has_valid_move
    mask[1] = true
  end
  
  return mask
end

valid_pos((col, row)) = 1 <= col <= BOARD_SIZE && 1 <= row <= BOARD_SIZE


count_pieces(b::Board, p::Player) = count(c -> c == p, b)


# Update statuses for `finished` and `winner`
initializing_board(b::Board) = count_pieces(b, EMPTY) > NUM_CELLS - 4

function update_status!(g::GameEnv)
  println("update_status!")
  if (g.initializing)
    g.initializing = initializing_board(g.board)
    g.finished = false
    g.winner = EMPTY
  else
    g.finished = any(1:64) do pos
      g.board[pos] != EMPTY || 
        is_valid_move(g.board, WHITE, pos) ||
        is_valid_move(g.board, BLACK, pos)
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
end


## Update game environment with chosen action (for current player).
function GI.play!(g::GameEnv, action)
  println("play!")
  g.curplayer = other(g.curplayer)
  if action != 0  # 'pass' action
    g.board = update_game_board_move(g.board, g.curplayer, index_to_pos(UInt8(action)))
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




const BASE_WIN_REWARD = 100.0
const PIECES_WIN_REWARD = 50.0
const PERFECT_WIN_REWARD = 50.0

## Returns the *immediate* reward obtained by the white player after last move.
# TODO: Consider better reward functions.
function GI.white_reward(g::GameEnv)
  if g.finished
    enemy_pieces = count_pieces(g.board, other(g.winner))
    reward =  BASE_WIN_REWARD
    reward += PIECES_WIN_REWARD * ((31 - enemy_pieces) / 31)
    reward += enemy_pieces == 0 ? PERFECT_WIN_REWARD : 0

    # Invert reward if black won
    reward = (g.winner == WHITE) ? reward : -reward
    return reward
  else
    return 0
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
=#


## Heuristic estimate of the state for *current player*.
## Not needed for AlphaZero, but is useful for baselines like minimax.
function GI.heuristic_value(g::GameEnv)
  p = g.curplayer
  my_count = count_pieces(g.board, p)
  enemy_count = count_pieces(g.board, other(p))

  return my_count - enemy_count
end


#####
##### ML interface
#####

flip_cell_color(c::Cell) = c == EMPTY ? EMPTY : other(c)

function flip_colors(board)
  return @SVector [
    flip_cell_color(board[pos])
    for col in 1:NUM_CELLS]
end


## Returns the vectorized version of a gamestate.
## Basically, an array of floats given to the neural network (I think)
function GI.vectorize_state(::GameSpec, state)
  board = state.curplayer == WHITE ? state.board : flip_colors(state.board)
  return Float32[
    board[pos_to_index(toPos((x, y)))] == c
    for y in 1:BOARD_SIZE,
        x in 1:BOARD_SIZE,
        c in [EMPTY, WHITE, BLACK]]
end

#####
##### Symmetries
#####

function generate_dihedral_symmetries()
  N = BOARD_SIZE
  rot((x, y)) = toPos((y, N - x + 1)) # 90° rotation
  flip((x, y)) = toPos((x, N - y + 1)) # flip along vertical axis
  ap(f) = p -> pos_to_index(f(index_to_pos(UInt8(p))))
  sym(f) = map(ap(f), collect(UInt8, 1:NUM_CELLS))
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
    "$a) Pass"
  else
    xy = index_to_pos(a)
    "$a) Play tile at $(col_letter(xy[1]))$(xy[2])"
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
      pos = pos_to_index((col, row))
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
