%## -*- texinfo -*-
%## @deftypefn {Function File} {} raft_image_write( @var{str}, @var{im}, @var{tl_x}, @var{tl_y}, @var{br_x}, @var{br_y} )
%## Writes the image samples stored at @var{im} in a file named @var{str}.
%## The image is stored in a format that can be read by 'raft_image_read'.
%
% The last four values will determine the image sampling interval (a rectangle
% given by its top-left and bottom-right corners) are optional and default to
% -1 1 -1 1, respectively.
% @seealso{raft_image_read}
% @end deftypefn
% Author: Elias Salomão Helou Neto <elias@icmc.usp.br>
% Created: 2013-10-21

function raft_image_write( str, im, tl_x, tl_y, br_x, br_y, value_type, size_type, order )

  [fid, msg] = fopen( str, 'w' );
  if fid < 0
    printf('Falha ao abrir o arquivo. Mensagem do sistema:\n%s\n',msg);
    return;
  end;

  if ( nargin < 3 )
    tl_x = -1;
  end
  if ( nargin < 4 )
    tl_y = 1;
  end
  if ( nargin < 5 )
    br_x = 1;
  end
  if ( nargin < 6 )
    br_y = -1;
  end

  if ( nargin < 7 )
    value_type = 'float32';
  end

  if ( nargin < 8 )
    size_type = 'int32';
  end

  if ( nargin < 9 )
    order = 'f';
  end



  [ m n ] = size( im );

  fwrite( fid, m, size_type );
  fwrite( fid, n, size_type );

  if strcmpi( order, 'c' ) == 1
    im = fwrite( fid, im', value_type );
  else if strcmpi( order, 'f' ) == 1
    im = fwrite( fid, im, value_type );
  else
    printf( 'Unknown specified order\n' );
    im = [];
    fclose( fid );
    return;
  end

  fwrite( fid, tl_x, value_type );
  fwrite( fid, tl_y, value_type );
  fwrite( fid, br_x, value_type );
  fwrite( fid, br_y, value_type );

  fclose( fid );

  end
