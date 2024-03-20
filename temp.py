def interpn(vol, loc, fill_value=None):
    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)
    nb_dims = loc.shape[-1]
    input_vol_shape = vol.shape

    if len(vol.shape) not in [nb_dims, nb_dims + 1]:
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    # flatten and float location Tensors
    if not loc.dtype.is_floating:
        target_loc_dtype = vol.dtype if vol.dtype.is_floating else 'float32'
        loc = tf.cast(loc, target_loc_dtype)
    elif vol.dtype.is_floating and vol.dtype != loc.dtype:
        loc = tf.cast(loc, vol.dtype)

    if isinstance(vol.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = vol.shape.as_list()
    else:
        volshape = vol.shape

    max_loc = [d - 1 for d in vol.get_shape().as_list()]

    # interpolate
    # floor has to remain floating-point since we will use it in such operation
    loc0 = tf.floor(loc)

    # clip values
    clipped_loc = [tf.clip_by_value(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
    loc0lst = [tf.clip_by_value(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

    # get other end of point cube
    loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
    locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

    # compute the difference between the upper value and the original value
    # differences are basically 1 - (pt - floor(pt))
    #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
    diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
    diff_loc0 = [1 - d for d in diff_loc1]
    # note reverse ordering since weights are inverse of diff.
    weights_loc = [diff_loc1, diff_loc0]

    # go through all the cube corners, indexed by a ND binary vector
    # e.g. [0, 0] means this "first" corner in a 2-D "cube"
    cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
    interp_vol = 0

    for c in cube_pts:
        subs = [locs[c[d]][d] for d in range(nb_dims)]
        idx = sub2ind2d(vol.shape[:-1], subs)
        vol_reshape = tf.reshape(vol, [-1, volshape[-1]])
        vol_val = tf.gather(vol_reshape, idx)
        wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
        wt = prod_n(wts_lst)
        wt = K.expand_dims(wt, -1)
        interp_vol += wt * vol_val

    if fill_value is not None:
        out_type = interp_vol.dtype
        fill_value = tf.constant(fill_value, dtype=out_type)
        below = [tf.less(loc[..., d], 0) for d in range(nb_dims)]
        above = [tf.greater(loc[..., d], max_loc[d]) for d in range(nb_dims)]
        out_of_bounds = tf.reduce_any(tf.stack(below + above, axis=-1), axis=-1, keepdims=True)
        interp_vol *= tf.cast(tf.logical_not(out_of_bounds), out_type)
        interp_vol += tf.cast(out_of_bounds, out_type) * fill_value

    # if only inputted volume without channels C, then return only that channel
    if len(input_vol_shape) == nb_dims:
        assert interp_vol.shape[-1] == 1, 'Something went wrong with interpn channels'
        interp_vol = interp_vol[..., 0]

    return interp_vol