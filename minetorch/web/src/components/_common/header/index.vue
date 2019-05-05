<template>
  <el-header class="header-content">
    <el-row>
      <el-col :span="6">
        <div class="grid-content title">minetorch</div>
      </el-col>
      <el-col :span="18">
        <div class="grid-content">
          <div class="user">
            <el-dropdown
              trigger="click"
              @command="handleCommand">
              <p>
                <img
                  class="user-avatar"
                  src="@/image/common/user.png"
                >
                admin
              </p>
            </el-dropdown>
          </div>
        </div>
      </el-col>
    </el-row>
  </el-header>
</template>

<script>
export default {
  methods: {
    handleCommand (cmd) {
      switch (cmd) {
        case 'signout':
          this.signout()
          break
        default:;
      }
    }
  }
}
</script>

<style lang="scss">
.header-content {
  background: #303133;
  .grid-content {
    height: 60px;
    line-height: 60px;
    color: #fff;
  }
  .title {
    font-size: 20px;
  }
  .user{
    text-align: right;
    font-size: 16px;
    p {
      cursor: pointer;
      color: #fff;
    }
  }
  .user-avatar {
    margin-right: 10px;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    vertical-align: middle;
  }
}
</style>
