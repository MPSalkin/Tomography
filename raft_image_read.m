%  -*- texinfo -*-
%  @deftypefn {Function File} [ @var{im} @var{tl_x}, @var{tl_y}, @var{br_x}, @var{br_y} ] = raft_image_read( @var{str}[, @var{value_type}][, @var{size_type}][, @var{order}] )
%  Reads the image from the file named @var{str}. @var{value_type} and @var{size_type}
%  are strings determining the types of floating point and integer values, respectively.
%  @var{order} can be 'c' or 'f', specifying row-major or column-major ordering of the input.
%  Defaults are:
%  @var{value_type} = 'double' and @var{size_type} = 'int32'.
% 
%  The last four output values determine the image sampling interval (a rectangle given by its top-left and bottom-right corners).
% @seealso{raft_image_write}
% @end deftypefn
% 
% Author: Elias Salomão Helou Neto <elias@icmc.usp.br>
% Created: 2013-10-21
% Modified: 2015-12-09
% Order of storage option corrected and documented.
% Modified: 2015-01-18
% Variable types for integers and floats;
% Error message translated to english.

function [ im tl_x tl_y br_x br_y ] = raft_image_read( str, value_type, size_type, order );

   if nargin < 2
      value_type = 'double';
   end

   if nargin < 3
      size_type = 'int32';
   end

   if nargin < 4
      order = 'f';
   end

  [ fid, msg ] = fopen( str, 'r' );
  if fid < 0
    printf( 'Failed to open file. System message:\n%s\n', msg );
    im = [];
    return;
  end;

  m = fread( fid, 1, size_type );
  n = fread( fid, 1, size_type );

  if strcmpi( order, 'c' ) == 1
    im = fread( fid, [n m], value_type )';
  else if strcmpi( order, 'f' ) == 1
    im = fread( fid, [m n], value_type );
  else
    printf( 'Unknown specified order\n' );
    im = [];
    fclose( fid );
    return;
  end

  tl_x = fread( fid, 1, value_type );
  tl_y = fread( fid, 1, value_type );
  br_x = fread( fid, 1, value_type );
  br_y = fread( fid, 1, value_type );

  fclose( fid );

end;
